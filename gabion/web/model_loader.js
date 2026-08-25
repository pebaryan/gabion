// model_loader.js
// Load a gabion wire-format model JSON (tools/export_model.py output) into a
// BBTTransformer, attaching a tokenizer and text-generation conveniences.
// Wire format: { config, weights_b64 (f16 base64), vocab?, merges? }
(function () {
  "use strict";

  async function loadBBTModel(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
    const data = await res.json();
    if (!data.weights_b64 || !data.config) throw new Error("not a gabion wire-format model");

    const weights = window.tinygradV0.f16Base64ToWeights(data.weights_b64);
    const c = data.config;
    const model = new window.tinygradV0.BBTTransformer({
      vocabSize: c.vocab_size, dModel: c.d_model, nHeads: c.n_heads,
      kvHeads: c.n_kv_heads || c.n_heads, nLayers: c.n_layers,
      seqLen: c.seq_len, dFF: c.d_ff,
      tieWeights: c.tie_weights, actQuant: c.act_quant, ropeBase: c.rope_base,
    });
    const consumed = model.loadFlatWeights(weights, false);
    if (consumed !== weights.length) {
      throw new Error(`weight cursor mismatch: consumed ${consumed} of ${weights.length}`);
    }
    if (data.vocab && data.merges && window.GPT2Tokenizer) {
      model.tokenizer = new window.GPT2Tokenizer({ vocab: data.vocab, merges: data.merges });
    }
    return attachGenerators(model);
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

  window.gabionLoader = { loadBBTModel, attachGenerators };
})();
