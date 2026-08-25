// Validate the GPT-2 byte-level BPE tokenizer against the real gpt2 vocab fixture
// plus a synthetic mini-BPE exercising the merge logic. Exit non-zero on failure.
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const sandbox = { window: {}, console, TextDecoder, TextEncoder, Math, Error, Int32Array, Array, Object, String, Number };
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root, "gabion/web/tokenizer.js"), "utf8"), sandbox, { filename: "tokenizer.js" });
const GPT2Tokenizer = sandbox.GPT2Tokenizer;

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) console.log(`PASS ${name}`);
  else { failures++; console.log(`FAIL ${name} ${detail}`); }
}

// ---- Synthetic mini BPE: verifies greedy merge ordering + byte fallback ----
{
  // Byte-level tokens for "ab": a(97), b(98); plus merge "a b" -> "ab" (rank 0)
  const a = String.fromCharCode(97), b = String.fromCharCode(98);
  const tok = new GPT2Tokenizer({
    vocab: { [a]: 0, [b]: 1, [a + b]: 2, [a + a + b]: 3 },
    merges: [`${a} ${b}`],
  });
  const ids = Array.from(tok.encode("ab"));
  check("mini encode merges to single token", ids.length === 1 && ids[0] === 2, JSON.stringify(ids));
  check("mini decode round-trip", tok.decode(ids) === "ab");
  // "aab": merge greedily left-to-right: (a b) -> ab, then (a ab) no rule -> [a, ab]
  const ids2 = Array.from(tok.encode("aab"));
  check("mini greedy left-to-right", ids2.length === 2 && ids2[0] === 0 && ids2[1] === 2, JSON.stringify(ids2));
}

// ---- Real GPT-2 tokenizer ----
{
  const fixture = JSON.parse(fs.readFileSync(path.join(here, "_gpt2_tok.json"), "utf8"));
  const tok = new GPT2Tokenizer({ vocab: fixture.vocab, merges: fixture.merges });
  check("gpt2 vocab size", tok.vocabSize === 50257, String(tok.vocabSize));

  const ids = Array.from(tok.encode("Hello world!"));
  check("gpt2 encode known ids", ids.length === 3 && ids[0] === 15496 && ids[1] === 995 && ids[2] === 0,
    JSON.stringify(ids));

  const roundTrip = tok.decode(ids);
  check("gpt2 decode round-trip", roundTrip === "Hello world!", JSON.stringify(roundTrip));

  // Multi-line / unicode / punctuation stress
  const stress = "The quick brown fox jumps over 42 dogs.\nÄpfel & 🍎 — 100%!";
  const stressIds = tok.encode(stress);
  const stressBack = tok.decode(stressIds);
  check("gpt2 unicode round-trip", stressBack === stress, JSON.stringify(stressBack));

  // '#'-prefixed merges ("# #" -> "##", "## ##" -> "####") are real BPE rules and
  // must be present in the fixture; dropping them silently splits "####" into pieces
  const hashIds = Array.from(tok.encode("####"));
  check("gpt2 '#' merges applied", hashIds.length === 1 && hashIds[0] === 4242, JSON.stringify(hashIds));

  // Idempotent: re-encoding the decoded text gives the same ids
  const reIds = Array.from(tok.encode(stressBack));
  check("gpt2 encode(decode(x)) stable", JSON.stringify(reIds) === JSON.stringify(Array.from(stressIds)));
}

if (failures > 0) {
  console.log(`\n${failures} failure(s)`);
  process.exit(2);
}
console.log("\nall tokenizer checks passed");
