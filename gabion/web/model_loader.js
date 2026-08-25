// model_loader.js
// Load a gabion wire-format model JSON (tools/export_model.py output) into a
// BBTTransformer, attaching a tokenizer and text-generation conveniences.
// Wire format: { config, weights_b64 (f16 base64), vocab?, merges? }
(function () {
  "use strict";

  /** Build a BBTTransformer from wire data (config + weights + optional tokenizer). */
  function buildModel(data, weights) {
    if (!data.config) throw new Error("not a gabion wire-format model");
    const c = data.config;
    const model = new window.tinygradV0.BBTTransformer({
      vocabSize: c.vocab_size, dModel: c.d_model, nHeads: c.n_heads,
      kvHeads: c.n_kv_heads || c.n_heads, nLayers: c.n_layers,
      seqLen: c.seq_len, dFF: c.d_ff,
      tieWeights: c.tie_weights, actQuant: c.act_quant, ropeBase: c.rope_base,
      qBiases: data.q_bias ? data.q_bias.map(a => new Float32Array(a)) : null,
      kBiases: data.k_bias ? data.k_bias.map(a => new Float32Array(a)) : null,
      vBiases: data.v_bias ? data.v_bias.map(a => new Float32Array(a)) : null,
    });
    const consumed = model.loadFlatWeights(weights, false);
    if (consumed !== weights.length) {
      throw new Error(`weight cursor mismatch: consumed ${consumed} of ${weights.length}`);
    }
    if (data.vocab && data.merges && window.GPT2Tokenizer) {
      model.tokenizer = new window.GPT2Tokenizer({ vocab: data.vocab, merges: data.merges, special: data.special });
    }
    model.chatTemplate = data.chat_template || null;
    return attachGenerators(model);
  }

  async function loadBBTModel(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
    const data = await res.json();
    if (!data.weights_b64) throw new Error("not a gabion wire-format model (weights_b64 missing)");
    const weights = window.tinygradV0.f16Base64ToWeights(data.weights_b64);
    return buildModel(data, weights);
  }

  /**
   * Binary wire: model.json (config/vocab/merges — no weights) + weights.f16
   * (raw little-endian f16 flat). Avoids parsing a multi-GB base64 JSON string:
   * the flat is fetched as an ArrayBuffer and converted via a half->float LUT.
   */
  async function loadBBTModelBin(modelJsonUrl, f16Url) {
    const [jres, fres] = await Promise.all([
      fetch(modelJsonUrl), fetch(f16Url),
    ]);
    if (!jres.ok) throw new Error(`fetch ${modelJsonUrl}: ${jres.status}`);
    if (!fres.ok) throw new Error(`fetch ${f16Url}: ${fres.status}`);
    const data = await jres.json();
    const buf = await fres.arrayBuffer();
    if (buf.byteLength % 2 !== 0) throw new Error("weights.f16 has odd byte length");
    const n = buf.byteLength / 2;
    const u16 = new Uint16Array(buf);
    const weights = new Float32Array(n);
    const lut = halfToFloatLUT();
    for (let i = 0; i < n; i++) weights[i] = lut[u16[i]];
    return buildModel(data, weights);
  }

  let _halfLUT = null;
  function halfToFloatLUT() {
    if (_halfLUT) return _halfLUT;
    const lut = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = (h & 0x8000) ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const m = h & 0x3ff;
      if (e === 0) lut[h] = s * m * 2 ** -24;
      else if (e === 31) lut[h] = m ? NaN : s * Infinity;
      else lut[h] = s * (1 + m / 1024) * 2 ** (e - 15);
    }
    _halfLUT = lut;
    return lut;
  }

  /** Attach tokenize / generateText conveniences to a BBTTransformer. */
  function attachGenerators(model) {
    if (model.tokenize) return model;
    model.tokenize = (text) => {
      if (model.tokenizer) return Array.from(model.tokenizer.encode(text));
      // byte-level fallback (BBT trains on raw bytes, vocab 256)
      const out = new Array(text.length);
      for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) & 0xFF;
      return out;
    };
    model.generateText = async (prompt, opts = {}) => {
      const ids = model.tokenize(prompt);
      const res = await model.decode(ids, opts);
      const text = model.tokenizer ? model.tokenizer.decode(res.tokens) : null;
      return { ...res, text };
    };
    return model;
  }

  window.gabionLoader = { loadBBTModel, loadBBTModelBin, attachGenerators };
})();
