// tokenizer.js
// GPT-2 byte-level BPE tokenizer (the de-facto standard for llama-family models).
// Loaded from { vocab: {token: id}, merges: ["a b", ...] } — the exact shape
// produced by tools/export_model.py --with-tokenizer gpt2.
(function () {
  "use strict";

  function bytesToUnicode() {
    const bs = [...Array(256).keys()];
    const cs = [...bs];
    let n = 0;
    for (let b = 0; b < 256; b++) {
      const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      if (printable) continue;
      cs[b] = 256 + n;
      n += 1;
    }
    const table = {};
    for (let b = 0; b < 256; b++) table[b] = String.fromCharCode(cs[b]);
    return table;
  }

  const BYTE_ENCODER = bytesToUnicode();
  const BYTE_DECODER = {};
  for (const [b, ch] of Object.entries(BYTE_ENCODER)) BYTE_DECODER[ch] = Number(b);

  // GPT-2 pre-tokenization regex (bytes: whitespace, letters, numbers, punctuation).
  const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  class GPT2Tokenizer {
    /**
     * @param {object} spec
     * @param {object|Array} spec.vocab - token -> id (JSON objects from HF keep order)
     * @param {string[]} spec.merges - merge rules in rank order ("a b")
     */
    constructor(spec) {
      if (!spec || !spec.vocab) throw new Error("GPT2Tokenizer needs {vocab, merges}");
      const vocab = Array.isArray(spec.vocab) ? spec.vocab : spec.vocab;
      this.vocab = {};
      this.idToToken = {};
      if (Array.isArray(vocab)) {
        for (let i = 0; i < vocab.length; i++) {
          this.vocab[vocab[i]] = i;
          this.idToToken[i] = vocab[i];
        }
      } else {
        for (const [tok, id] of Object.entries(vocab)) {
          this.vocab[tok] = id;
          this.idToToken[id] = tok;
        }
      }
      this.ranks = {};
      if (spec.merges) {
        for (let i = 0; i < spec.merges.length; i++) this.ranks[spec.merges[i]] = i;
      }
    }

    get vocabSize() {
      return Object.keys(this.vocab).length;
    }

    /** Encode text to token ids (Int32Array). */
    encode(text) {
      const ids = [];
      for (const m of String(text).matchAll(PAT)) {
        const piece = m[0];
        // Map the UTF-8 BYTES of the piece through the byte->unicode table
        // (code units would break on surrogate pairs / non-ASCII).
        const pieceBytes = new TextEncoder().encode(piece);
        let chars = "";
        for (let i = 0; i < pieceBytes.length; i++) chars += BYTE_ENCODER[pieceBytes[i]];
        const parts = [];
        for (let i = 0; i < chars.length; i++) parts.push(chars[i]);
        // Greedy byte-pair merge
        while (parts.length > 1) {
          let best = null;
          for (let i = 0; i < parts.length - 1; i++) {
            const pair = parts[i] + " " + parts[i + 1];
            const rank = this.ranks[pair];
            if (rank !== undefined && (best === null || rank < best.rank)) best = { rank, i };
          }
          if (!best) break;
          parts[best.i] = parts[best.i] + parts[best.i + 1];
          parts.splice(best.i + 1, 1);
        }
        for (const p of parts) {
          const id = this.vocab[p];
          if (id === undefined) throw new Error(`tokenizer: token "${p}" not in vocab`);
          ids.push(id);
        }
      }
      return Int32Array.from(ids);
    }

    /** Decode token ids back to text. */
    decode(ids) {
      let text = "";
      for (const id of ids) {
        const token = this.idToToken[id];
        if (token === undefined) continue;
        text += token;
      }
      const bytes = new Uint8Array(text.length);
      for (let i = 0; i < text.length; i++) {
        const code = text.charCodeAt(i);
        const b = BYTE_DECODER[text[i]];
        bytes[i] = b === undefined ? code & 0xFF : b;
      }
      return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
    }
  }

  window.GPT2Tokenizer = GPT2Tokenizer;
})();
