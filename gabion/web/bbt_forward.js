// bbt_forward.js
// Full BBT (BitByte Transformer) forward/backward pass in JS.
// Mirrors gabion/user_models/bbt_transformer.py with optional BitLinear quantization.
(function () {
  "use strict";

  const { Tensor, crossEntropy, rmsNorm } = window.tinygradV0;

  /** Check if WebGPU backend is available. */
  function gpu() {
    return window.WebGPUBackend && WebGPUBackend.instance;
  }

  class BBTTransformer {
    /**
     * @param {object} config
     * @param {number} config.vocabSize  - default 256
     * @param {number} config.dModel    - default 64
     * @param {number} config.nHeads    - default 4
     * @param {number} config.nLayers   - default 2
     * @param {number} config.seqLen    - default 32
     * @param {number} config.dFF       - default dModel*4
     * @param {boolean} config.tieWeights - default true
     * @param {number} config.ropeBase  - default 10000
     */
    constructor(config = {}) {
      this.V = config.vocabSize || 256;
      this.D = config.dModel || 64;
      this.H = config.nHeads || 4;
      this.L = config.nLayers || 2;
      this.T = config.seqLen || 32;
      this.dFF = config.dFF || (this.D * 4);
      this.tieWeights = config.tieWeights !== false;
      this.ropeBase = config.ropeBase || 10000.0;
      this.headDim = this.D / this.H;
      this.eps = 1e-6;
      this.actQuant = config.actQuant !== false; // default true

      // Precompute RoPE inverse frequencies: [headDim/2]
      const halfDim = this.headDim / 2;
      this._invFreq = new Float32Array(halfDim);
      for (let i = 0; i < halfDim; i++) {
        this._invFreq[i] = Math.exp(-(Math.log(this.ropeBase) / this.headDim) * (i * 2));
      }

      // Precompute cos/sin tables for max seq_len: [T, headDim]
      this._cosTable = new Float32Array(this.T * this.headDim);
      this._sinTable = new Float32Array(this.T * this.headDim);
      for (let t = 0; t < this.T; t++) {
        for (let i = 0; i < halfDim; i++) {
          const freq = t * this._invFreq[i];
          const c = Math.cos(freq);
          const s = Math.sin(freq);
          // Duplicate: [cos(f0), cos(f1), ..., cos(f0), cos(f1), ...]
          this._cosTable[t * this.headDim + i] = c;
          this._cosTable[t * this.headDim + halfDim + i] = c;
          this._sinTable[t * this.headDim + i] = s;
          this._sinTable[t * this.headDim + halfDim + i] = s;
        }
      }

      // GPU buffers for RoPE tables (lazy-initialized)
      this._ropeCosBuf = null;
      this._ropeSinBuf = null;
    }

    /** Upload RoPE cos/sin tables to GPU (once). */
    _ensureRopeGPU() {
      const backend = gpu();
      if (!backend) return false;
      if (!this._ropeCosBuf) {
        this._ropeCosBuf = backend.createBufferFromData(this._cosTable);
        this._ropeSinBuf = backend.createBufferFromData(this._sinTable);
      }
      return true;
    }

    /** Release GPU RoPE buffers. */
    releaseGPU() {
      if (this._ropeCosBuf) { this._ropeCosBuf.destroy(); this._ropeCosBuf = null; }
      if (this._ropeSinBuf) { this._ropeSinBuf.destroy(); this._ropeSinBuf = null; }
    }

    /**
     * Deserialize flat weight array into param Tensors.
     * Layout matches Python BBTTransformerAdapter.init_params.
     *
     * Returns array of Tensor objects with requiresGrad=true.
     */
    deserializeParams(flatWeights, uploadToGPU = false) {
      const params = [];
      let cursor = 0;

      const take = (shape, gpuUpload = false) => {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = flatWeights.subarray(cursor, cursor + size);
        cursor += size;
        const t = Tensor.fromArray(data, shape, true);
        if (gpuUpload) t.toGPU();
        return t;
      };

      // tok_emb [V, D] — uploaded to GPU (used in final projection matmul)
      params.push(take([this.V, this.D], uploadToGPU));

      // Per layer: q_w, k_w, v_w, o_w [D,D], n1_w [D], gate_up_w [D, 2*dFF], n2_w [D], down_w [dFF, D]
      for (let l = 0; l < this.L; l++) {
        params.push(take([this.D, this.D], uploadToGPU));       // q_w — GPU
        params.push(take([this.D, this.D], uploadToGPU));       // k_w — GPU
        params.push(take([this.D, this.D], uploadToGPU));       // v_w — GPU
        params.push(take([this.D, this.D], uploadToGPU));       // o_w — GPU
        params.push(take([this.D]));                            // n1_w — CPU (1D, used in rmsNorm)
        params.push(take([this.D, 2 * this.dFF], uploadToGPU)); // gate_up_w — GPU
        params.push(take([this.D]));                            // n2_w — CPU (1D)
        params.push(take([this.dFF, this.D], uploadToGPU));     // down_w — GPU
      }

      // norm_f_w [D] — CPU (1D)
      params.push(take([this.D]));

      // Optional lm_head_w [D, V] — GPU
      if (!this.tieWeights) {
        params.push(take([this.D, this.V], uploadToGPU));
      }

      return params;
    }

    /**
     * Serialize param Tensors back to flat Float32Array.
     */
    serializeParams(params) {
      let totalSize = 0;
      for (const p of params) totalSize += p.numel;
      const out = new Float32Array(totalSize);
      let cursor = 0;
      for (const p of params) {
        out.set(p.data, cursor);
        cursor += p.numel;
      }
      return out;
    }

    /**
     * Forward pass: x [B, T] token ids → logits [B, T-1, V].
     * @param {Tensor[]} params - from deserializeParams
     * @param {Int32Array} xFlat - flattened [B*T] token indices
     * @param {number} B - batch size
     * @param {number} T - sequence length
     */
    async forward(params, xFlat, B, T, ternarize = false) {
      let idx = 0;
      const tokEmb = params[idx++];

      // Embedding lookup: [B*T] -> [B, T, D]
      const hasGPUBackend = !!(window.WebGPUBackend && WebGPUBackend.instance);
      let x = hasGPUBackend
        ? Tensor.embeddingLookup2DGPU(tokEmb, xFlat, B, T)
        : Tensor.embeddingLookup2D(tokEmb, xFlat, B, T);

      // Transformer blocks
      for (let l = 0; l < this.L; l++) {
        const qW = params[idx++];
        const kW = params[idx++];
        const vW = params[idx++];
        const oW = params[idx++];
        const n1W = params[idx++];
        const gateUpW = params[idx++];
        const n2W = params[idx++];
        const downW = params[idx++];
        x = await this._block(x, B, T, qW, kW, vW, oW, n1W, gateUpW, n2W, downW, ternarize);
      }

      // Final norm (CPU op — ensure x is on CPU)
      if (x.onGPU) await x.toCPU();
      const normFW = params[idx++];
      x = rmsNorm(x, normFW, this.eps);

      // LM head projection: [B, T, D] @ [D, V] -> [B, T, V]
      let lmWeight;
      if (this.tieWeights) {
        lmWeight = tokEmb;  // [V, D] -> need transpose
      } else {
        lmWeight = params[idx++];
      }

      // Reshape x to [B*T, D], matmul with [D, V] (GPU-accelerated), reshape to [B, T, V]
      const xFlat2d = x.reshape([B * T, this.D]);
      let logitsW;
      if (this.tieWeights) {
        logitsW = lmWeight.transpose2d();  // [V,D] -> [D,V]
      } else {
        logitsW = lmWeight;  // already [D, V]
      }
      const logitsFlat = xFlat2d.matmul(logitsW);  // [B*T, V] — may be on GPU
      // Readback logits for CPU cross-entropy
      if (logitsFlat.onGPU) await logitsFlat.toCPU();
      const logits3d = logitsFlat.reshape([B, T, this.V]);

      // Slice off last position: predict x[1..T] from x[0..T-1]
      // logits[:, :T-1, :] -> flatten to [(B*(T-1)), V]
      const outT = T - 1;
      const outData = new Float32Array(B * outT * this.V);
      for (let b = 0; b < B; b++) {
        for (let t = 0; t < outT; t++) {
          const srcOff = (b * T + t) * this.V;
          const dstOff = (b * outT + t) * this.V;
          for (let v = 0; v < this.V; v++) {
            outData[dstOff + v] = logits3d.data[srcOff + v];
          }
        }
      }

      // Create a tensor that routes gradients back to logits3d
      return new Tensor(outData, [B * outT, this.V], logits3d.requiresGrad, [logits3d], (gout) => {
        if (!logits3d.requiresGrad) return;
        if (!logits3d.grad) logits3d.grad = new Float32Array(logits3d.data.length);
        for (let b = 0; b < B; b++) {
          for (let t = 0; t < outT; t++) {
            const srcOff = (b * outT + t) * this.V;
            const dstOff = (b * T + t) * this.V;
            for (let v = 0; v < this.V; v++) {
              logits3d.grad[dstOff + v] += gout[srcOff + v];
            }
          }
        }
      });
    }

    /**
     * Transformer block: pre-norm attention + pre-norm FFN with residuals.
     * x: [B, T, D]
     */
    async _block(x, B, T, qW, kW, vW, oW, n1W, gateUpW, n2W, downW, ternarize = false) {
      // Pre-norm (CPU) + attention (GPU matmuls + CPU ops) + residual
      if (x.onGPU) await x.toCPU();
      let h = rmsNorm(x, n1W, this.eps);
      h = await this._causalSelfAttention(h, B, T, qW, kW, vW, oW, ternarize);
      if (h.onGPU) await h.toCPU();
      x = x.add(h);

      // Pre-norm (CPU) + FFN (GPU matmuls + CPU ops) + residual
      if (x.onGPU) await x.toCPU();
      h = rmsNorm(x, n2W, this.eps);
      h = await this._swiGLU(h, B, T, gateUpW, downW, ternarize);
      if (h.onGPU) await h.toCPU();
      x = x.add(h);

      return x;
    }

    /**
     * Multi-head causal self-attention with RoPE.
     * x: [B, T, D] -> [B, T, D]
     */
    async _causalSelfAttention(x, B, T, qW, kW, vW, oW, ternarize = false) {
      const D = this.D;
      const H = this.H;
      const headDim = this.headDim;
      const BH = B * H;
      const backend = gpu();

      // Project: [B*T, D] @ [D, D] -> [B*T, D]
      const x2d = x.reshape([B * T, D]);
      let q, k, v;
      if (ternarize) {
        q = this._bitlinear(x2d, qW);
        k = this._bitlinear(x2d, kW);
        v = this._bitlinear(x2d, vW);
      } else {
        q = x2d.matmul(qW);
        k = x2d.matmul(kW);
        v = x2d.matmul(vW);
      }

      // Readback Q/K/V for CPU reshape (head reordering)
      if (q.onGPU) await q.toCPU();
      if (k.onGPU) await k.toCPU();
      if (v.onGPU) await v.toCPU();

      // Reshape to [B*H, T, headDim] (CPU — data reordering)
      q = this._reshapeForHeads(q, B, T, H, headDim);
      k = this._reshapeForHeads(k, B, T, H, headDim);
      v = this._reshapeForHeads(v, B, T, H, headDim);

      // GPU-accelerated attention core (RoPE + scores + softmax + weighted sum)
      // Forward outputs are kept on GPU for the backward pass.
      if (backend && this._ensureRopeGPU()) {
        // Upload reshaped Q/K/V to GPU
        const qBuf = backend.createBufferFromData(q.data);
        const kBuf = backend.createBufferFromData(k.data);
        const vBuf = backend.createBufferFromData(v.data);

        // Batch all forward attention dispatches into a single submit
        backend.beginBatch();
        // RoPE on GPU: [BH, T, headDim]
        const qRopeBuf = backend.rope(qBuf, this._ropeCosBuf, this._ropeSinBuf, BH, T, headDim);
        const kRopeBuf = backend.rope(kBuf, this._ropeCosBuf, this._ropeSinBuf, BH, T, headDim);
        // Fused attention: Q·K^T * scale + causal mask + softmax → attn [BH, T, T]
        const attnBuf = backend.fusedAttention(qRopeBuf, kRopeBuf, BH, T, headDim);
        // Weighted sum: attn @ V → [BH, T, headDim]
        const yBuf = backend.batchedMatmul(attnBuf, vBuf, BH, T, T, headDim);
        backend.endBatch();

        backend.releaseBuffer(qBuf);
        backend.releaseBuffer(kBuf);

        // Read back result for CPU reshape
        const yData = await backend.readBuffer(yBuf, BH * T * headDim);
        backend.releaseBuffer(yBuf);

        // Capture GPU buffers and model ref for backward closure
        const savedAttnBuf = attnBuf;
        const savedQRopeBuf = qRopeBuf;
        const savedKRopeBuf = kRopeBuf;
        const savedVBuf = vBuf;
        const ropeCosBuf = this._ropeCosBuf;
        const ropeSinBuf = this._ropeSinBuf;
        const scale = 1.0 / Math.sqrt(headDim);

        const yTensor = new Tensor(yData, [BH, T, headDim], q.requiresGrad, [q, k, v], (gout) => {
          // GPU backward for attention
          if (!q.requiresGrad && !k.requiresGrad && !v.requiresGrad) {
            backend.releaseBuffer(savedAttnBuf); backend.releaseBuffer(savedQRopeBuf);
            backend.releaseBuffer(savedKRopeBuf); backend.releaseBuffer(savedVBuf);
            return;
          }

          const goutBuf = backend.createBufferFromData(gout);

          // Batch all backward attention dispatches into a single submit
          backend.beginBatch();

          // dAttn = gout @ V^T  [BH, T, T]
          const dAttnBuf = backend.batchedMatmul(goutBuf, savedVBuf, BH, T, headDim, T, true);

          // dV = attn^T @ gout  [BH, T, headDim]
          let attnTBuf = null;
          if (v.requiresGrad) {
            attnTBuf = backend.batchedTranspose(savedAttnBuf, BH, T, T);
            const dVBuf = backend.batchedMatmul(attnTBuf, goutBuf, BH, T, T, headDim);
            v._pendingGradBuf = dVBuf;
          }

          // dScores = softmaxBackward(attn, dAttn) * scale  [BH*T, T]
          const dScoresBuf = backend.softmaxBackward(savedAttnBuf, dAttnBuf, BH * T, T, scale);

          // dQrope = dScores @ Krope  [BH, T, headDim]
          const dQropeBuf = backend.batchedMatmul(dScoresBuf, savedKRopeBuf, BH, T, T, headDim);

          // dKrope = dScores^T @ Qrope  [BH, T, headDim]
          const dScoresTBuf = backend.batchedTranspose(dScoresBuf, BH, T, T);
          const dKropeBuf = backend.batchedMatmul(dScoresTBuf, savedQRopeBuf, BH, T, T, headDim);

          // RoPE backward: dQ = ropeBackward(dQrope), dK = ropeBackward(dKrope)
          let dQBuf = null, dKBuf = null;
          if (q.requiresGrad) {
            dQBuf = backend.ropeBackward(dQropeBuf, ropeCosBuf, ropeSinBuf, BH, T, headDim);
          }
          if (k.requiresGrad) {
            dKBuf = backend.ropeBackward(dKropeBuf, ropeCosBuf, ropeSinBuf, BH, T, headDim);
          }

          backend.endBatch();

          // Assign pending grad buffers
          if (dQBuf) q._pendingGradBuf = dQBuf;
          if (dKBuf) k._pendingGradBuf = dKBuf;

          // Release intermediate buffers back to pool
          backend.releaseBuffer(goutBuf);
          backend.releaseBuffer(dAttnBuf);
          if (attnTBuf) backend.releaseBuffer(attnTBuf);
          backend.releaseBuffer(dScoresBuf);
          backend.releaseBuffer(dScoresTBuf);
          backend.releaseBuffer(dQropeBuf);
          backend.releaseBuffer(dKropeBuf);
          backend.releaseBuffer(savedAttnBuf);
          backend.releaseBuffer(savedQRopeBuf);
          backend.releaseBuffer(savedKRopeBuf);
          backend.releaseBuffer(savedVBuf);
        });

        // Reshape back: [B*H, T, headDim] -> [B, T, D]
        const y2 = this._reshapeFromHeads(yTensor, B, T, H, headDim);

        // Output projection
        const y2d = y2.reshape([B * T, D]);
        const out2d = ternarize ? this._bitlinear(y2d, oW) : y2d.matmul(oW);
        if (out2d.onGPU) await out2d.toCPU();
        return out2d.reshape([B, T, D]);
      }

      // --- CPU fallback path ---

      // Apply RoPE to q and k (CPU)
      q = this._applyRoPE(q, BH, T, headDim);
      k = this._applyRoPE(k, BH, T, headDim);

      // Attention scores: [B*H, T, headDim] @ [B*H, headDim, T] -> [B*H, T, T]
      const kT = k.transpose3d();
      let scores = q.batchedMatmul(kT);
      scores = scores.scale(1.0 / Math.sqrt(headDim));

      // Causal mask + softmax
      scores = scores.causalMask();
      const attn = scores.softmax();

      // Weighted sum: [B*H, T, T] @ [B*H, T, headDim] -> [B*H, T, headDim]
      let y = attn.batchedMatmul(v);

      // Reshape back: [B*H, T, headDim] -> [B, T, D]
      y = this._reshapeFromHeads(y, B, T, H, headDim);

      // Output projection: [B*T, D] @ [D, D] -> [B*T, D]
      const y2d = y.reshape([B * T, D]);
      const out2d = ternarize ? this._bitlinear(y2d, oW) : y2d.matmul(oW);
      if (out2d.onGPU) await out2d.toCPU();
      return out2d.reshape([B, T, D]);
    }

    /**
     * Reshape [B*T, D] -> [B*H, T, headDim].
     * Conceptually: [B*T, D] -> [B, T, H, headDim] -> [B, H, T, headDim] -> [B*H, T, headDim]
     */
    _reshapeForHeads(x, B, T, H, headDim) {
      // x: [B*T, D] where D = H * headDim
      // We need to reorder to [B*H, T, headDim]
      const D = H * headDim;
      const out = new Float32Array(B * H * T * headDim);
      for (let b = 0; b < B; b++) {
        for (let t = 0; t < T; t++) {
          for (let h = 0; h < H; h++) {
            const srcOff = (b * T + t) * D + h * headDim;
            const dstOff = (b * H + h) * T * headDim + t * headDim;
            for (let d = 0; d < headDim; d++) {
              out[dstOff + d] = x.data[srcOff + d];
            }
          }
        }
      }
      return new Tensor(out, [B * H, T, headDim], x.requiresGrad, [x], (gout) => {
        if (!x.requiresGrad) return;
        if (!x.grad) x.grad = new Float32Array(x.data.length);
        for (let b = 0; b < B; b++) {
          for (let t = 0; t < T; t++) {
            for (let h = 0; h < H; h++) {
              const dstOff = (b * T + t) * D + h * headDim;
              const srcOff = (b * H + h) * T * headDim + t * headDim;
              for (let d = 0; d < headDim; d++) {
                x.grad[dstOff + d] += gout[srcOff + d];
              }
            }
          }
        }
      });
    }

    /**
     * Reshape [B*H, T, headDim] -> [B, T, D] (inverse of _reshapeForHeads).
     */
    _reshapeFromHeads(x, B, T, H, headDim) {
      const D = H * headDim;
      const out = new Float32Array(B * T * D);
      for (let b = 0; b < B; b++) {
        for (let t = 0; t < T; t++) {
          for (let h = 0; h < H; h++) {
            const srcOff = (b * H + h) * T * headDim + t * headDim;
            const dstOff = (b * T + t) * D + h * headDim;
            for (let d = 0; d < headDim; d++) {
              out[dstOff + d] = x.data[srcOff + d];
            }
          }
        }
      }
      return new Tensor(out, [B, T, D], x.requiresGrad, [x], (gout) => {
        if (!x.requiresGrad) return;
        if (!x.grad) x.grad = new Float32Array(x.data.length);
        for (let b = 0; b < B; b++) {
          for (let t = 0; t < T; t++) {
            for (let h = 0; h < H; h++) {
              const dstOff = (b * H + h) * T * headDim + t * headDim;
              const srcOff = (b * T + t) * D + h * headDim;
              for (let d = 0; d < headDim; d++) {
                x.grad[dstOff + d] += gout[srcOff + d];
              }
            }
          }
        }
      });
    }

    /**
     * Ternary weight quantization (BitNet-style).
     * Scales weights by mean(|w|), quantizes to {-1, 0, 1}, rescales.
     * Uses Straight-Through Estimator for gradient flow.
     * w: any shape -> same shape
     */
    _ternaryQuantBitnet(w, ste = false) {
      const gamma = w.abs().meanAll(); // scalar
      // Clamp gamma to eps for stability
      const gammaVal = Math.max(gamma.data[0], this.eps);
      const wScaled = w.scale(1.0 / gammaVal);
      const wqHard = wScaled.clip(-1.0, 1.0).round();
      let wq;
      if (ste) {
        // STE: forward uses quantized, backward uses identity
        // wq = wScaled + (wqHard - wScaled).detach()
        // Since detach() stops gradient flow through the difference,
        // forward value = wScaled + (wqHard - wScaled) = wqHard
        // backward gradient flows through wScaled only
        const diff = new Tensor(
          new Float32Array(wqHard.data.length),
          [...wqHard.shape], false, [], null
        );
        for (let i = 0; i < diff.data.length; i++) {
          diff.data[i] = wqHard.data[i] - wScaled.data[i];
        }
        wq = wScaled.add(diff); // diff is detached, grad flows through wScaled
      } else {
        wq = wqHard;
      }
      return wq.scale(gammaVal);
    }

    /**
     * Per-token activation quantization.
     * Quantizes activations to [-127, 127] per row (token).
     * Uses STE for gradient flow.
     * x: [rows, D] or [B, T, D] -> same shape
     */
    _actQuantPerToken(x, q = 127) {
      if (!this.actQuant) return x;
      const shape = x.shape;
      const D = shape[shape.length - 1];
      const rows = x.numel / D;

      // Reshape to 2D for per-row scaling
      const x2d = x.reshape([rows, D]);
      // Per-row max(|x|) / q
      const xAbs = x2d.abs();
      const sRaw = xAbs.maxLastAxis(); // [rows, 1], no grad
      // Clamp scaling factor
      const sData = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        sData[r] = Math.max(sRaw.data[r] / q, this.eps);
      }
      const s = new Tensor(sData, [rows, 1], false, [], null);

      // Scale, quantize, dequantize with STE
      const xScaled = x2d.div(s);
      const xqHard = xScaled.clip(-q, q).round();
      // STE: forward = xqHard, backward flows through xScaled
      const diff = new Tensor(
        new Float32Array(xqHard.data.length),
        [...xqHard.shape], false, [], null
      );
      for (let i = 0; i < diff.data.length; i++) {
        diff.data[i] = xqHard.data[i] - xScaled.data[i];
      }
      const xq = xScaled.add(diff);

      // Multiply back by scale factor and reshape
      const out2d = xq.mul(new Tensor(
        (() => {
          const d = new Float32Array(rows * D);
          for (let r = 0; r < rows; r++) {
            for (let j = 0; j < D; j++) d[r * D + j] = sData[r];
          }
          return d;
        })(),
        [rows, D], false, [], null
      ));

      return out2d.reshape(shape);
    }

    /**
     * BitLinear: activation quantization + weight quantization + matmul.
     * x: [N, D_in], w: [D_in, D_out] -> [N, D_out]
     */
    _bitlinear(x, w, quantizedWeights = false) {
      x = this._actQuantPerToken(x);
      let wq;
      if (quantizedWeights) {
        wq = w; // already quantized
      } else {
        wq = this._ternaryQuantBitnet(w, true);
      }
      return x.matmul(wq);
    }

    /**
     * Apply Rotary Position Embeddings.
     * x: [BH, T, headDim] -> [BH, T, headDim]
     */
    _applyRoPE(x, BH, T, headDim) {
      const halfDim = headDim / 2;
      const out = new Float32Array(x.data.length);
      const cosT = this._cosTable;
      const sinT = this._sinTable;

      for (let bh = 0; bh < BH; bh++) {
        for (let t = 0; t < T; t++) {
          const off = (bh * T + t) * headDim;
          const tOff = t * headDim;
          for (let i = 0; i < halfDim; i++) {
            const x1 = x.data[off + i];
            const x2 = x.data[off + halfDim + i];
            const c = cosT[tOff + i];
            const s = sinT[tOff + i];
            // RoPE: rotate_half(x) = [-x2, x1]
            // result = x * cos + rotate_half(x) * sin
            out[off + i] = x1 * c - x2 * s;
            out[off + halfDim + i] = x2 * c + x1 * s;
          }
        }
      }

      return new Tensor(out, [...x.shape], x.requiresGrad, [x], (gout) => {
        if (!x.requiresGrad) return;
        if (!x.grad) x.grad = new Float32Array(x.data.length);
        for (let bh = 0; bh < BH; bh++) {
          for (let t = 0; t < T; t++) {
            const off = (bh * T + t) * headDim;
            const tOff = t * headDim;
            for (let i = 0; i < halfDim; i++) {
              const c = cosT[tOff + i];
              const s = sinT[tOff + i];
              const g1 = gout[off + i];
              const g2 = gout[off + halfDim + i];
              // Backward of rotation:
              // dx1 = g1*c + g2*s, dx2 = -g1*s + g2*c
              x.grad[off + i] += g1 * c + g2 * s;
              x.grad[off + halfDim + i] += -g1 * s + g2 * c;
            }
          }
        }
      });
    }

    /**
     * Apply RoPE on raw Float32Array data (for backward recomputation).
     * Returns a new Float32Array.
     */
    _applyRoPEData(data, BH, T, headDim) {
      const halfDim = headDim / 2;
      const out = new Float32Array(data.length);
      const cosT = this._cosTable;
      const sinT = this._sinTable;
      for (let bh = 0; bh < BH; bh++) {
        for (let t = 0; t < T; t++) {
          const off = (bh * T + t) * headDim;
          const tOff = t * headDim;
          for (let i = 0; i < halfDim; i++) {
            const c = cosT[tOff + i], s = sinT[tOff + i];
            out[off + i] = data[off + i] * c - data[off + halfDim + i] * s;
            out[off + halfDim + i] = data[off + halfDim + i] * c + data[off + i] * s;
          }
        }
      }
      return out;
    }

    /**
     * SwiGLU FFN: gate_up projection, split, silu(gate) * up, down projection.
     * x: [B, T, D]
     */
    async _swiGLU(x, B, T, gateUpW, downW, ternarize = false) {
      const D = this.D;
      const dFF = this.dFF;
      const N = B * T;
      const backend = (window.WebGPUBackend && WebGPUBackend.instance) || null;

      // [B*T, D] @ [D, 2*dFF] -> [B*T, 2*dFF]
      const x2d = x.reshape([N, D]);
      const gateUp = ternarize
        ? this._bitlinear(x2d, gateUpW)
        : x2d.matmul(gateUpW);

      if (backend && gateUp.onGPU && !ternarize) {
        // GPU path: fused silu(gate) * up on GPU, skip CPU readback
        const gateUpBuf = gateUp.gpuBuffer;
        const activatedBuf = backend.siluMul(gateUpBuf, N, dFF);

        // Create activated tensor with GPU buffer for the down matmul
        const activated = new Tensor(new Float32Array(N * dFF), [N, dFF], true, [gateUp], (gout) => {
          // Backward: compute dGateUp from dOut and original gateUp
          if (!gateUp.requiresGrad) return;
          // This backward is handled by the GPU path in resolveGradsGPU
          if (activated._pendingGradBuf && gateUpBuf) {
            const dABuf = backend.siluMulBackward(activated._pendingGradBuf, gateUpBuf, N, dFF);
            gateUp._pendingGradBuf = dABuf;
            backend.releaseBuffer(activated._pendingGradBuf);
            activated._pendingGradBuf = null;
          }
        });
        activated.gpuBuffer = activatedBuf;
        activated._dirty = "gpu";

        // Down projection: [B*T, dFF] @ [dFF, D] -> [B*T, D]
        const out2d = activated.matmul(downW);
        if (out2d.onGPU) await out2d.toCPU();
        return out2d.reshape([B, T, D]);
      }

      // CPU fallback
      if (gateUp.onGPU) await gateUp.toCPU();
      const [gate, up] = gateUp.splitLast([dFF, dFF]);
      const activated = gate.silu().mul(up);

      const out2d = ternarize
        ? this._bitlinear(activated, downW)
        : activated.matmul(downW);
      if (out2d.onGPU) await out2d.toCPU();
      return out2d.reshape([B, T, D]);
    }
  }

  /**
   * Full BBT transformer training function.
   * Called from browser worker's handleRoundStart.
   *
   * @param {Float32Array} weights - flat weight vector from mesh
   * @param {object} opts - training options
   * @returns {{ updated: Float32Array, loss: number, sampleCount: number, mode: string }}
   */
  async function trainLocalV1(weights, opts) {
    const V = opts.vocabSize || 256;
    const D = opts.dModel || 64;
    const H = opts.nHeads || 4;
    const L = opts.nLayers || 2;
    const T = opts.seqLen || 32;
    const dFF = opts.dFF || (D * 4);
    const tieWeights = opts.tieWeights !== false;

    const ternarize = !!opts.ternarize;
    const actQuant = opts.actQuant !== false; // default true when ternarize

    const model = new BBTTransformer({
      vocabSize: V, dModel: D, nHeads: H, nLayers: L,
      seqLen: T, dFF: dFF, tieWeights: tieWeights,
      actQuant: ternarize ? actQuant : false, // only quantize activations when ternarizing
    });

    // Check if weights match expected size
    const expectedSize = model._expectedParamCount();
    if (weights.length < expectedSize) {
      // Weights don't match full transformer — fall back to v0
      return window.tinygradV0.trainLocalV0(weights, opts);
    }

    const hasGPU = !!(window.WebGPUBackend && WebGPUBackend.instance);
    const params = model.deserializeParams(weights, hasGPU);

    const lr = Math.max(1e-7, opts.lr || 5e-4);
    const epochs = Math.max(1, opts.epochs || 1);
    const batchSize = Math.max(2, opts.batchSize || 8);
    let seed = opts.seed >>> 0;

    let lastLoss = 0.0;
    let samples = 0;
    const backend = hasGPU ? WebGPUBackend.instance : null;

    // Allocate GPU Adam state buffers (zero-initialized, persist across epochs)
    const useAdam = (opts.optimizer || "adam") === "adam";
    if (backend && useAdam) {
      for (const p of params) {
        const bytes = p.numel * 4;
        p._adamMBuf = backend.createEmptyBuffer(bytes);
        p._adamVBuf = backend.createEmptyBuffer(bytes);
        backend.writeBuffer(p._adamMBuf, new Float32Array(p.numel));
        backend.writeBuffer(p._adamVBuf, new Float32Array(p.numel));
      }
    }

    // Pre-compute epoch data (cheap Int32Array allocations, enables double-buffering)
    const seqBatch = (opts.sequences && Array.isArray(opts.sequences) && opts.sequences.length > 0) ? opts.sequences : null;
    function prepEpochData(e) {
      let xFlat, yFlat, actualB;
      if (seqBatch && seqBatch.length >= batchSize) {
        actualB = Math.min(batchSize, seqBatch.length);
        xFlat = new Int32Array(actualB * T);
        yFlat = new Int32Array(actualB * (T - 1));
        for (let b = 0; b < actualB; b++) {
          const seq = seqBatch[b];
          for (let t = 0; t < T; t++) xFlat[b * T + t] = (seq[t] | 0) % V;
          for (let t = 0; t < T - 1; t++) yFlat[b * (T - 1) + t] = (seq[t + 1] | 0) % V;
        }
      } else {
        actualB = batchSize;
        xFlat = new Int32Array(actualB * T);
        yFlat = new Int32Array(actualB * (T - 1));
        let s = (seed ^ (e * 2654435761)) >>> 0;
        const rnd = () => { s = (Math.imul(1664525, s) + 1013904223) >>> 0; return s; };
        for (let b = 0; b < actualB; b++) {
          for (let t = 0; t < T; t++) xFlat[b * T + t] = rnd() % V;
          for (let t = 0; t < T - 1; t++) yFlat[b * (T - 1) + t] = xFlat[b * T + t + 1];
        }
      }
      return { xFlat, yFlat, actualB };
    }

    // Pre-prepare first epoch data; subsequent epochs prepared during GPU compute
    let nextData = prepEpochData(0);

    const profiling = !!opts.profile && backend && backend._hasTimestamps;
    if (profiling) backend.enableProfiling();

    for (let e = 0; e < epochs; e++) {
      if (profiling) backend.resetProfiling();

      // Zero grads
      for (const p of params) {
        p.grad = null;
        if (p._gradGPUBuf && backend) { backend.releaseBuffer(p._gradGPUBuf); p._gradGPUBuf = null; }
      }

      const { xFlat, yFlat, actualB } = nextData;
      // Pre-prepare next epoch's data (overlaps with GPU compute below)
      if (e + 1 < epochs) nextData = prepEpochData(e + 1);

      // Forward (async — GPU matmuls with CPU readbacks)
      const logits = await model.forward(params, xFlat, actualB, T, ternarize);  // [B*(T-1), V]
      const loss = backend
        ? await crossEntropyGPU(logits, yFlat)
        : crossEntropy(logits, yFlat);

      // Backward
      loss.backward();

      // --- Optimizer step ---
      const useAdam = (opts.optimizer || "adam") === "adam";
      const beta1 = opts.adamBeta1 || 0.9;
      const beta2 = opts.adamBeta2 || 0.999;
      const epsAdam = 1e-8;
      window._adamStep = (window._adamStep || 0) + 1;
      const astep = window._adamStep;
      const warmup = Math.max(1, opts.warmupSteps || 10);
      const effLr = astep <= warmup ? lr * (astep / warmup) : lr;
      const bc1 = 1 - Math.pow(beta1, astep);
      const bc2 = 1 - Math.pow(beta2, astep);
      const maxNorm = opts.gradClipNorm != null ? opts.gradClipNorm : 1.0;

      if (backend) {
        // --- GPU path: grads stay on GPU, optimizer via compute shader ---
        await loss.resolveGradsGPU();

        // Gradient clipping: reduce sumSquares per param, read scalars, conditional scale
        if (maxNorm > 0) {
          backend.beginBatch();
          const normBufs = params.map(p =>
            p._gradGPUBuf ? backend.reduce(p._gradGPUBuf, 1, p.numel, 2) : null
          );
          backend.endBatch();
          const normSqs = await Promise.all(
            normBufs.map(buf => buf
              ? backend.readBuffer(buf, 1).then(d => { backend.releaseBuffer(buf); return d[0]; })
              : Promise.resolve(0)
            )
          );
          const totalNorm = Math.sqrt(normSqs.reduce((a, b) => a + b, 0));
          if (totalNorm > maxNorm) {
            const clipScale = maxNorm / totalNorm;
            backend.beginBatch();
            const scaledBufs = params.map(p => {
              if (!p._gradGPUBuf) return null;
              return backend.elementwise(p._gradGPUBuf, null, p.numel, 3, clipScale);
            });
            backend.endBatch();
            params.forEach((p, i) => {
              if (scaledBufs[i]) {
                backend.releaseBuffer(p._gradGPUBuf);
                p._gradGPUBuf = scaledBufs[i];
              }
            });
          }
        }

        // Optimizer update dispatch
        backend.beginBatch();
        if (useAdam) {
          for (const p of params) {
            if (!p._gradGPUBuf) continue;
            backend.adamUpdate(
              p._gradGPUBuf, p._adamMBuf, p._adamVBuf, p.gpuBuffer,
              p.numel, beta1, beta2, effLr, bc1, bc2, epsAdam
            );
          }
        } else {
          for (const p of params) {
            if (!p._gradGPUBuf) continue;
            backend.sgdUpdate(p._gradGPUBuf, p.gpuBuffer, p.numel, lr);
          }
        }
        backend.endBatch();

        // Release grad buffers
        for (const p of params) {
          if (p._gradGPUBuf) { backend.releaseBuffer(p._gradGPUBuf); p._gradGPUBuf = null; }
        }

      } else {
        // --- CPU path: resolve grads to CPU, clip + Adam/SGD on CPU ---
        if (hasGPU) await loss.resolveGrads();

        if (maxNorm > 0) {
          let totalNormSq = 0;
          for (const p of params) {
            if (p.grad) for (let i = 0; i < p.grad.length; i++) totalNormSq += p.grad[i] * p.grad[i];
          }
          const totalNorm = Math.sqrt(totalNormSq);
          if (totalNorm > maxNorm) {
            const clipScale = maxNorm / totalNorm;
            for (const p of params) {
              if (p.grad) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= clipScale;
            }
          }
        }

        if (useAdam) {
          for (const p of params) {
            if (!p.grad) continue;
            if (!p._adam_m) p._adam_m = new Float32Array(p.data.length);
            if (!p._adam_v) p._adam_v = new Float32Array(p.data.length);
            const m = p._adam_m, v = p._adam_v, g = p.grad;
            for (let i = 0; i < p.data.length; i++) {
              m[i] = beta1 * m[i] + (1 - beta1) * g[i];
              v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
              p.data[i] -= effLr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + epsAdam);
            }
            if (typeof p.markCPUDirty === "function") p.markCPUDirty();
          }
        } else {
          for (const p of params) {
            if (p.grad) {
              for (let i = 0; i < p.data.length; i++) p.data[i] -= lr * p.grad[i];
              if (typeof p.markCPUDirty === "function") p.markCPUDirty();
            }
          }
        }

        // Re-upload updated weights to GPU for next epoch
        for (const p of params) {
          if (hasGPU && p.gpuBuffer) p.toGPU();
        }
      }

      // Read and log profiling results for this epoch
      if (profiling) {
        const timings = await backend.readProfilingResults();
        if (timings.length > 0) {
          const summary = timings.map(t => `${t.label}: ${t.durationMs.toFixed(3)}ms`).join(", ");
          const total = timings.reduce((s, t) => s + t.durationMs, 0);
          console.log(`[profile] epoch ${e}: ${summary} | total GPU: ${total.toFixed(3)}ms`);
        }
      }

      lastLoss = loss.data[0];
      samples += actualB * (T - 1);
    }

    if (profiling) backend.disableProfiling();

    // Read final weights back from GPU after all epochs
    if (backend) {
      await Promise.all(params.map(p => p.toCPU()));
    }

    // Release GPU buffers + Adam state
    if (hasGPU) {
      for (const p of params) {
        if (p._adamMBuf && backend) { backend.releaseBuffer(p._adamMBuf); p._adamMBuf = null; }
        if (p._adamVBuf && backend) { backend.releaseBuffer(p._adamVBuf); p._adamVBuf = null; }
        if (p._gradGPUBuf && backend) { backend.releaseBuffer(p._gradGPUBuf); p._gradGPUBuf = null; }
        p.releaseGPU();
      }
      model.releaseGPU(); // RoPE tables
    }

    // Re-serialize
    const updated = model.serializeParams(params);

    const qTag = ternarize ? "-bitlinear" : "";
    const gpuTag = hasGPU ? "-webgpu" : "";
    const f16Tag = opts.useF16 ? "-f16" : "";
    const mode = `v1-bbt-transformer${qTag}${gpuTag}${f16Tag}`;

    // If original weights were longer (shouldn't happen but be safe), preserve tail
    let finalWeights = updated;
    if (weights.length > updated.length) {
      finalWeights = new Float32Array(weights.length);
      finalWeights.set(updated);
      finalWeights.set(weights.subarray(updated.length), updated.length);
    }

    // Encode weights for transfer: int8 delta > f16 base64 > plain JSON
    const result = { loss: Number(lastLoss), sampleCount: samples, mode };
    if (window.tinygradV0.encodeInt8Delta) {
      const enc = window.tinygradV0.encodeInt8Delta(finalWeights, weights);
      result.weights_delta = enc.delta_b64;
      result.delta_scale = enc.delta_scale;
      result.weights_format = "int8_delta";
      result.weights_count = enc.count;
    } else if (opts.useF16 && window.tinygradV0.weightsToF16Base64) {
      result.weights_f16 = window.tinygradV0.weightsToF16Base64(finalWeights);
      result.weights_format = "f16_base64";
      result.weights_count = finalWeights.length;
    } else {
      result.updated = finalWeights;
    }
    return result;
  }

  // Add expected param count helper
  BBTTransformer.prototype._expectedParamCount = function () {
    let count = this.V * this.D;  // tok_emb
    for (let l = 0; l < this.L; l++) {
      count += 4 * this.D * this.D;    // q, k, v, o
      count += this.D;                  // n1_w
      count += this.D * 2 * this.dFF;  // gate_up
      count += this.D;                  // n2_w
      count += this.dFF * this.D;       // down
    }
    count += this.D;  // norm_f_w
    if (!this.tieWeights) count += this.D * this.V;
    return count;
  };

  // Attach to global
  window.tinygradV0.BBTTransformer = BBTTransformer;
  window.tinygradV0.trainLocalV1 = trainLocalV1;
})();
