// bbt_forward.js
// Full BBT (BitByte Transformer) forward/backward pass in JS.
// Mirrors gabion/user_models/bbt_transformer.py with optional BitLinear quantization.
(function () {
  "use strict";

  const { Tensor, Optimizer, nn, kvAttention } = window.tinygradV0;

  /** Single transformer block: pre-norm attention + pre-norm FFN with residuals. */
  class TransformerBlock extends nn.Module {
    constructor(D, H, kvD, dFF) {
      super();
      this.q = new nn.Linear(D, D);
      this.k = new nn.Linear(D, kvD);
      this.v = new nn.Linear(D, kvD);
      this.o = new nn.Linear(D, D);
      this.norm1 = new nn.RMSNorm(D);
      this.gateUp = new nn.Linear(D, 2 * dFF);
      this.norm2 = new nn.RMSNorm(D);
      this.down = new nn.Linear(dFF, D);
    }
  }

  class BBTTransformer extends nn.Module {
    /**
     * @param {object} config
     * @param {number} config.vocabSize  - default 256
     * @param {number} config.dModel    - default 64
     * @param {number} config.nHeads    - default 4
     * @param {number} config.kvHeads   - default nHeads (MHA); GQA when fewer
     * @param {number} config.nLayers   - default 2
     * @param {number} config.seqLen    - default 32
     * @param {number} config.dFF       - default dModel*4
     * @param {boolean} config.tieWeights - default true
     * @param {number} config.ropeBase  - default 10000
     */
    constructor(config = {}) {
      super();
      this.V = config.vocabSize || 256;
      this.D = config.dModel || 64;
      this.H = config.nHeads || 4;
      this.kvH = config.kvHeads || config.nHeads || 4;
      this.L = config.nLayers || 2;
      this.T = config.seqLen || 32;
      this.dFF = config.dFF || (this.D * 4);
      this.tieWeights = config.tieWeights !== false;
      this.ropeBase = config.ropeBase || 10000.0;
      this.headDim = this.D / this.H;
      this.kvD = this.kvH * this.headDim;
      this.eps = 1e-6;
      this.actQuant = config.actQuant !== false; // default true
      // Optional attention biases (q/k/v per-layer; Qwen2.5-Instruct has them)
      this._qBiases = config.qBiases || null;
      this._kBiases = config.kBiases || null;
      this._vBiases = config.vBiases || null;

      // nn layers
      this.tokEmb = new nn.Embedding(this.V, this.D);
      this.layers = [];
      for (let l = 0; l < this.L; l++) {
        const bl = new TransformerBlock(this.D, this.H, this.kvD, this.dFF);
        if (this._qBiases) bl.qBias = this._qBiases[l];
        if (this._kBiases) bl.kBias = this._kBiases[l];
        if (this._vBiases) bl.vBias = this._vBiases[l];
        this.layers.push(bl);
      }
      this.normF = new nn.RMSNorm(this.D);
      if (!this.tieWeights) {
        this.lmHead = new nn.Linear(this.D, this.V);
      }

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
          this._cosTable[t * this.headDim + i] = c;
          this._cosTable[t * this.headDim + halfDim + i] = c;
          this._sinTable[t * this.headDim + i] = s;
          this._sinTable[t * this.headDim + halfDim + i] = s;
        }
      }

      // GPU buffers for RoPE tables (lazy-initialized, non-parameter)
      this._ropeCosBuf = null;
      this._ropeSinBuf = null;
    }

    /** Upload RoPE cos/sin tables to GPU (once). */
    _ensureRopeGPU() {
      const backend = window.WebGPUBackend && WebGPUBackend.instance;
      if (!backend) return false;
      if (!this._ropeCosBuf) {
        this._ropeCosBuf = backend.createBufferFromData(this._cosTable);
        this._ropeSinBuf = backend.createBufferFromData(this._sinTable);
      }
      return true;
    }

    /** Release GPU buffers (parameters + RoPE tables). */
    releaseGPU() {
      super.releaseGPU();
      if (this._ropeCosBuf) { this._ropeCosBuf.destroy(); this._ropeCosBuf = null; }
      if (this._ropeSinBuf) { this._ropeSinBuf.destroy(); this._ropeSinBuf = null; }
    }

    /**
     * Load from flat weight array (server format).
     * Layout: tok_emb, [per-layer: q,k,v,o, n1, gate_up, n2, down], norm_f, [lm_head]
     * Compatible with Python BBTTransformerAdapter.init_params ordering.
     */
    loadFlatWeights(flatWeights, uploadToGPU = false) {
      let cursor = 0;
      const take = (tensor, gpuUpload = false) => {
        const size = tensor.numel;
        tensor.data.set(flatWeights.subarray(cursor, cursor + size));
        tensor.markCPUDirty();
        if (gpuUpload) tensor.toGPU();
        cursor += size;
      };

      take(this.tokEmb.weight, uploadToGPU);
      for (let l = 0; l < this.L; l++) {
        const bl = this.layers[l];
        take(bl.q.weight, uploadToGPU);
        take(bl.k.weight, uploadToGPU);
        take(bl.v.weight, uploadToGPU);
        take(bl.o.weight, uploadToGPU);
        take(bl.norm1.weight);          // CPU (1D, used in rmsNorm)
        take(bl.gateUp.weight, uploadToGPU);
        take(bl.norm2.weight);          // CPU (1D)
        take(bl.down.weight, uploadToGPU);
      }
      take(this.normF.weight);
      if (!this.tieWeights) {
        take(this.lmHead.weight, uploadToGPU);
      }
      return cursor;
    }

    /** Serialize all parameters back to flat Float32Array (server format). */
    toFlatWeights() {
      const parts = [];
      parts.push(this.tokEmb.weight.data);
      for (let l = 0; l < this.L; l++) {
        const bl = this.layers[l];
        parts.push(bl.q.weight.data);
        parts.push(bl.k.weight.data);
        parts.push(bl.v.weight.data);
        parts.push(bl.o.weight.data);
        parts.push(bl.norm1.weight.data);
        parts.push(bl.gateUp.weight.data);
        parts.push(bl.norm2.weight.data);
        parts.push(bl.down.weight.data);
      }
      parts.push(this.normF.weight.data);
      if (!this.tieWeights) parts.push(this.lmHead.weight.data);

      let total = 0;
      for (const p of parts) total += p.length;
      const out = new Float32Array(total);
      let cursor = 0;
      for (const p of parts) { out.set(p, cursor); cursor += p.length; }
      return out;
    }

    /**
     * Forward pass: x [B, T] token ids -> logits [B, T-1, V].
     * @param {Int32Array} xFlat - flattened [B*T] token indices
     * @param {number} B - batch size
     * @param {number} T - sequence length
     */
    async forward(xFlat, B, T, ternarize = false) {
      // Embedding lookup: [B*T] -> [B, T, D] (auto-dispatches to GPU or CPU)
      let x = this.tokEmb.forward(xFlat, B, T);

      // Transformer blocks
      for (let l = 0; l < this.L; l++) {
        const bl = this.layers[l];
        x = await this._block(x, B, T, bl, ternarize);
      }

      // Final norm (rmsNorm forward is CPU, auto-readbacks if needed)
      if (x.onGPU) await x.toCPU();
      x = this.normF.forward(x);

      // LM head projection: [B, T, D] @ [D, V] -> [B, T, V]
      // Reshape x to [B*T, D], matmul with [D, V] (GPU-accelerated), reshape to [B, T, V]
      const xFlat2d = x.reshape([B * T, this.D]);
      let logitsW;
      if (this.tieWeights) {
        logitsW = this.tokEmb.weight.transpose2d();  // [V,D] -> [D,V]
      } else {
        logitsW = this.lmHead.weight;  // already [D, V]
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
     * x: [B, T, D], bl: TransformerBlock
     */
    async _block(x, B, T, bl, ternarize = false) {
      // Pre-norm + attention + residual
      if (x.onGPU) await x.toCPU(); // rmsNorm forward is CPU
      let h = bl.norm1.forward(x);
      h = await this._causalSelfAttention(h, B, T, bl.q.weight, bl.k.weight, bl.v.weight, bl.o.weight, ternarize, bl.qBias || null, bl.kBias || null, bl.vBias || null);
      if (h.onGPU) await h.toCPU();
      x = x.add(h);

      // Pre-norm + FFN + residual
      if (x.onGPU) await x.toCPU(); // rmsNorm forward is CPU
      h = bl.norm2.forward(x);
      h = await this._swiGLU(h, B, T, bl.gateUp.weight, bl.down.weight, ternarize);
      if (h.onGPU) await h.toCPU();
      x = x.add(h);

      return x;
    }

    /**
     * Multi-head causal self-attention with RoPE.
     * x: [B, T, D] -> [B, T, D]
     */
    async _causalSelfAttention(x, B, T, qW, kW, vW, oW, ternarize = false, qBias = null, kBias = null, vBias = null) {
      const D = this.D;
      const H = this.H;
      const headDim = this.headDim;
      const BH = B * H;
      const backend = window.WebGPUBackend && WebGPUBackend.instance;

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

      // Attention biases (Qwen2.5-Instruct): q/k/v projections are affine
      if (qBias) for (let i = 0; i < q.data.length; i++) q.data[i] += qBias[i % qBias.length];
      if (kBias) for (let i = 0; i < k.data.length; i++) k.data[i] += kBias[i % kBias.length];
      if (vBias) for (let i = 0; i < v.data.length; i++) v.data[i] += vBias[i % vBias.length];

      // Reshape to [B*H, T, headDim] (CPU — data reordering); k/v are grouped
      // when GQA (kvH < H) and expanded to full query-head count for the kernels.
      q = this._reshapeForHeads(q, B, T, H, headDim);
      k = this._reshapeForHeads(k, B, T, this.kvH, headDim);
      v = this._reshapeForHeads(v, B, T, this.kvH, headDim);
      k = this._expandKV(k, B, T);
      v = this._expandKV(v, B, T);

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
     * GQA: expand a KV-head tensor [B*kvH, T, headDim] to full query-head count
     * [B*H, T, headDim] by repeating each KV head `group` times in query-head
     * order (query head h of batch b attends to KV head h//group).
     * MHA (kvH === H) is a no-op returning x unchanged. The backward node
     * group-sums the incoming grad back to [B*kvH, T, headDim] (CPU loop or
     * the kv_group_sum kernel when the grad arrives as a GPU buffer).
     */
    _expandKV(x, B, T) {
      const kvH = this.kvH, H = this.H, headDim = this.headDim;
      if (kvH === H) return x;
      const group = H / kvH;
      const TD = T * headDim;
      const out = new Float32Array(B * H * TD);
      const src = x.data;
      // Query head h = kvh*group + g attends KV head kvh (h//group, contiguous
      // grouping — matches the Python adapter and the kv_attention kernels).
      for (let b = 0; b < B; b++) {
        for (let kvh = 0; kvh < kvH; kvh++) {
          for (let g = 0; g < group; g++) {
            out.set(src.subarray((b * kvH + kvh) * TD, (b * kvH + kvh + 1) * TD),
              (b * H + kvh * group + g) * TD);
          }
        }
      }
      const backend = window.WebGPUBackend && WebGPUBackend.instance;
      return new Tensor(out, [B * H, T, headDim], x.requiresGrad, [x], (gout, goutBuf) => {
        if (!x.requiresGrad) return;
        if (goutBuf && backend) {
          x._pendingGradBuf = backend.kvGroupSum(goutBuf, B, H, kvH, T, headDim);
          return;
        }
        if (!x.grad) x.grad = new Float32Array(x.data.length);
        for (let b = 0; b < B; b++) {
          for (let kvh = 0; kvh < kvH; kvh++) {
            const dstOff = (b * kvH + kvh) * TD;
            for (let g = 0; g < group; g++) {
              const srcOff = (b * H + g * kvH + kvh) * TD;
              for (let i = 0; i < TD; i++) x.grad[dstOff + i] += gout[srcOff + i];
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
    _applyRoPE(x, BH, T, headDim, startPos = 0) {
      const halfDim = headDim / 2;
      const out = new Float32Array(x.data.length);
      const cosT = this._cosTable;
      const sinT = this._sinTable;

      for (let bh = 0; bh < BH; bh++) {
        for (let t = 0; t < T; t++) {
          const off = (bh * T + t) * headDim;
          const tOff = (startPos + t) * headDim; // absolute position
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

    /**
     * Allocate a KV-cache state for autoregressive decode.
     * Cache tensors are [H, maxLen, headDim]; on GPU when a backend is present.
     * Returns { kCaches, vCaches, pos, maxLen }.
     */
    initKVCache(maxLen = null) {
      const backend = window.WebGPUBackend && WebGPUBackend.instance;
      const len = maxLen || this.T;
      const kvH = this.kvH;
      const headDim = this.headDim;
      const kCaches = [], vCaches = [];
      for (let l = 0; l < this.L; l++) {
        const kT = new Tensor(new Float32Array(kvH * len * headDim), [kvH, len, headDim], false, [], () => {},
          backend ? backend.createEmptyBuffer(kvH * len * headDim * 4) : null, backend ? "gpu" : "cpu");
        const vT = new Tensor(new Float32Array(kvH * len * headDim), [kvH, len, headDim], false, [], () => {},
          backend ? backend.createEmptyBuffer(kvH * len * headDim * 4) : null, backend ? "gpu" : "cpu");
        kCaches.push(kT);
        vCaches.push(vT);
      }
      return { kCaches, vCaches, pos: 0, maxLen: len };
    }

    /**
     * Autoregressive decode of one token (predicts the token AFTER tokenId given the
     * cached prefix). tokenId: number, state: from initKVCache().
     * Returns { logits: Float32Array[V], state }.
     * GPU path: RoPE at absolute position, k/v appended to the cache, attention via
     * the KV-cache kernels (O(L) per step). CPU fallback: re-runs the prefix forward
     * and takes the last predicted row — identical semantics, so the two paths are
     * interchangeable and the fallback doubles as a reference.
     */
    async decodeStep(tokenId, state, ternarize = false, debugNaN = false) {
      const D = this.D, H = this.H, V = this.V, headDim = this.headDim, BH = H, kvH = this.kvH;
      const backend = window.WebGPUBackend && WebGPUBackend.instance;
      const pos = state.pos;
      if (pos >= state.maxLen) throw new Error(`KV cache full (maxLen=${state.maxLen})`);
      const chk = (stage, buf) => {
        if (!debugNaN || !buf) return;
        for (let i = 0; i < buf.length; i++) if (!Number.isFinite(buf[i])) {
          throw new Error(`NaN at pos=${pos} ${stage}`);
        }
      };

      // CPU fallback: prefix forward, last predicted row (exact reference semantics)
      if (!backend) {
        if (!state._prefix) state._prefix = [];
        state._prefix.push(tokenId);
        const T = pos + 2;
        const logits = await this.forward(new Int32Array(state._prefix), 1, T, ternarize);
        const row = logits.data.subarray((T - 2) * V, (T - 1) * V);
        state.pos = pos + 1;
        return { logits: Float32Array.from(row), state };
      }

      // --- GPU KV-cache path ---
      this._ensureRopeGPU();
      let x = this.tokEmb.forward(new Int32Array([tokenId]), 1, 1);
      if (x.onGPU) await x.toCPU();

      for (let l = 0; l < this.L; l++) {
        const bl = this.layers[l];
        // Pre-norm + q/k/v projections
        let h = bl.norm1.forward(x);
        const h2d = h.reshape([1, D]);
        let q, k, v;
        if (ternarize) {
          q = this._bitlinear(h2d, bl.q.weight);
          k = this._bitlinear(h2d, bl.k.weight);
          v = this._bitlinear(h2d, bl.v.weight);
        } else {
          q = h2d.matmul(bl.q.weight);
          k = h2d.matmul(bl.k.weight);
          v = h2d.matmul(bl.v.weight);
        }
        if (q.onGPU) await q.toCPU();
        if (k.onGPU) await k.toCPU();
        if (v.onGPU) await v.toCPU();
        chk("layer" + l + ".q", q.data); chk("layer" + l + ".k", k.data); chk("layer" + l + ".v", v.data);
        // Attention biases (Qwen2.5-Instruct): affine projections
        if (bl.qBias) for (let i = 0; i < q.data.length; i++) q.data[i] += bl.qBias[i % bl.qBias.length];
        if (bl.kBias) for (let i = 0; i < k.data.length; i++) k.data[i] += bl.kBias[i % bl.kBias.length];
        if (bl.vBias) for (let i = 0; i < v.data.length; i++) v.data[i] += bl.vBias[i % bl.vBias.length];

        // Heads + RoPE at the absolute position (k/v have kvH heads under GQA)
        q = this._reshapeForHeads(q, 1, 1, H, headDim);
        k = this._reshapeForHeads(k, 1, 1, kvH, headDim);
        v = this._reshapeForHeads(v, 1, 1, kvH, headDim);
        q = this._applyRoPE(q, BH, 1, headDim, pos);
        k = this._applyRoPE(k, kvH, 1, headDim, pos);
        if (q.onGPU) await q.toCPU();
        if (k.onGPU) await k.toCPU();
        chk("layer" + l + ".q_rope", q.data); chk("layer" + l + ".k_rope", k.data);

        // Append k/v into the cache at position pos ([kvH, maxLen, headDim] layout)
        const kCache = state.kCaches[l], vCache = state.vCaches[l];
        for (let kv = 0; kv < kvH; kv++) {
          backend.writeBufferAt(kCache.gpuBuffer, k.data.subarray(kv * headDim, (kv + 1) * headDim), (kv * state.maxLen + pos) * headDim);
          backend.writeBufferAt(vCache.gpuBuffer, v.data.subarray(kv * headDim, (kv + 1) * headDim), (kv * state.maxLen + pos) * headDim);
        }

        // Attend over the prefix (0..pos) via the KV-cache kernels (GQA: query
        // head bh attends to cache row (bh * kvH) / H — mapped inside the kernels).
        // The cache is [kvH, maxLen, headDim] with slots > pos uninitialized, so
        // causal=1 masks j > pos to -inf BEFORE any cache read (kernels do this);
        // L stays maxLen (it doubles as the cache row stride in the kernels).
        const qT = Tensor.fromArray(q.data, [BH, headDim], false).toGPU();
        const y = kvAttention(qT, kCache, vCache, { causal: true, pos, kvH, H });
        // toCPU() returns the Tensor (data updated in place) — take .data
        const yData = (await y.toCPU()).data;
        chk("layer" + l + ".attn", yData);

        // Reshape back, output projection, residual
        const y3 = this._reshapeFromHeads(Tensor.fromArray(yData, [BH, 1, headDim], false), 1, 1, H, headDim);
        if (debugNaN) {
          if (!Number.isFinite(y3.data[0])) {
            throw new Error(`NaN at pos=${pos} layer${l}.o y3[0..3]=${Array.from(y3.data.slice(0, 4))} yData[0..3]=${Array.from(yData.slice(0, 4))} yDataLen=${yData.length} bh=${BH} hd=${headDim} y3len=${y3.data.length}`);
          }
        }
        const o = (ternarize ? this._bitlinear(y3.reshape([1, D]), bl.o.weight) : y3.reshape([1, D]).matmul(bl.o.weight)).reshape([1, 1, D]);
        if (o.onGPU) await o.toCPU();
        chk("layer" + l + ".o", o.data);
        x = x.add(o);
        chk("layer" + l + ".x", x.data);

        // FFN + residual
        let h2 = bl.norm2.forward(x);
        h2 = await this._swiGLU(h2, 1, 1, bl.gateUp.weight, bl.down.weight, ternarize);
        if (h2.onGPU) await h2.toCPU();
        chk("layer" + l + ".h2", h2.data);
        x = x.add(h2);
        chk("layer" + l + ".x2", x.data);
      }

      // Final norm + LM head
      x = this.normF.forward(x);
      const x2d = x.reshape([1, D]);
      const logitsW = this.tieWeights ? this.tokEmb.weight.transpose2d() : this.lmHead.weight;
      const logitsT = x2d.matmul(logitsW);
      if (logitsT.onGPU) await logitsT.toCPU();
      chk("final.logits", logitsT.data);
      state.pos = pos + 1;
      return { logits: Float32Array.from(logitsT.data), state };
    }

    /** Sample from logits with temperature + optional top-k/top-p (argmax when temperature <= 0).
     *  rng: () => [0,1); topP: nucleus threshold applied after top-k. */
    _sample(logits, temperature = 1.0, topK = 0, topP = 0, rng = Math.random) {
      const n = logits.length;
      if (temperature <= 0 || n <= 1) {
        let best = 0;
        for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
      }
      const scaled = new Float32Array(n);
      const invT = 1.0 / temperature;
      let maxV = -Infinity;
      for (let i = 0; i < n; i++) { scaled[i] = logits[i] * invT; if (scaled[i] > maxV) maxV = scaled[i]; }
      if (topK > 0 && topK < n) {
        const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => scaled[b] - scaled[a]);
        const cutoff = scaled[idx[topK - 1]];
        for (let i = 0; i < n; i++) if (scaled[i] < cutoff) scaled[i] = -Infinity;
      }
      if (topP > 0 && topP < 1) {
        // nucleus: keep the smallest set of tokens whose cumulative prob >= topP
        const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => scaled[b] - scaled[a]);
        let sumAll = 0;
        for (let i = 0; i < n; i++) sumAll += Math.exp(scaled[i] - maxV);
        let cum = 0;
        for (let j = 0; j < n; j++) {
          const e = Math.exp(scaled[idx[j]] - maxV);
          cum += e / sumAll;
          if (j > 0 && cum >= topP) { for (let jj = j; jj < n; jj++) scaled[idx[jj]] = -Infinity; break; }
        }
      }
      let sumE = 0;
      for (let i = 0; i < n; i++) sumE += Math.exp(scaled[i] - maxV);
      let r = Math.random() * sumE;
      for (let i = 0; i < n; i++) {
        r -= Math.exp(scaled[i] - maxV);
        if (r <= 0) return i;
      }
      return n - 1;
    }

    /**
     * Autoregressive generation from a prompt.
     * opts: { temperature=1.0, topK=0, topP=0, stopTokens=[], maxNewTokens=64, maxLen,
     *         ternarize=false, state, onToken, rng }
     * Sampling stops when a token in stopTokens is sampled (it stays in `tokens`)
     * or the context fills. Returns { tokens, logits, state, stopped }.
     */
    async decode(tokenIds, opts = {}) {
      const maxNew = opts.maxNewTokens != null ? opts.maxNewTokens : 64;
      const temperature = opts.temperature != null ? opts.temperature : 1.0;
      const topK = opts.topK || 0;
      const topP = opts.topP || 0;
      const stopTokens = opts.stopTokens || [];
      const rng = opts.rng || Math.random;
      const ternarize = !!opts.ternarize;
      const state = opts.state || this.initKVCache(opts.maxLen);
      const tokens = [...tokenIds];
      const logitsList = [];
      let stopped = false;
      // Prime the cache / prefix with the FULL prompt first: decodeStep sees one
      // token at a time, so every prompt token except the last must be fed before
      // generation starts (the last prompt token is consumed by the first step).
      for (let i = 0; i < tokens.length - 1; i++) {
        if (state.pos >= state.maxLen) break; // context full
        await this.decodeStep(tokens[i], state, ternarize, !!opts.debugNaN);
      }
      for (let i = 0; i < maxNew; i++) {
        if (state.pos >= state.maxLen) break; // context full
        const { logits, state: st } = await this.decodeStep(tokens[tokens.length - 1], state, ternarize, !!opts.debugNaN);
        const next = this._sample(logits, temperature, topK, topP, rng);
        tokens.push(next);
        logitsList.push(logits);
        if (opts.onToken) opts.onToken(next, logits, tokens);
        if (stopTokens.indexOf(next) !== -1) { stopped = true; break; }
      }
      return { tokens, logits: logitsList, state, stopped };
    }
  }

  // --- Utility functions for v0 training (BBT-specific) ---

  const { crossEntropy, embeddingLookup, rmsNorm } = window.tinygradV0;

  function gpu() {
    return window.WebGPUBackend && WebGPUBackend.instance;
  }

  function sampleBatch(vocabSize, batchSize, seed) {
    const x = new Int32Array(batchSize);
    const y = new Int32Array(batchSize);
    let s = seed >>> 0;
    function rnd() {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      return s;
    }
    for (let i = 0; i < batchSize; i++) {
      const t = rnd() % vocabSize;
      x[i] = t;
      y[i] = (t + 1) % vocabSize;
    }
    return { x, y };
  }

  function trainLocalV0(weights, opts) {
    const V = opts.vocabSize || 256;
    const D = opts.dModel || 64;
    const embBlock = V * D;

    const lr = Math.max(1e-7, opts.lr || 5e-4);
    const epochs = Math.max(1, opts.epochs || 1);
    const batchSize = Math.max(8, opts.batchSize || 64);
    let seed = opts.seed >>> 0;

    let lastLoss = 0.0;
    let samples = 0;

    if (weights.length >= embBlock) {
      const eView = weights.subarray(0, embBlock);
      const E = Tensor.fromArray(eView, [V, D], true);

      for (let e = 0; e < epochs; e++) {
        let batch = null;
        if (opts.batch && opts.batch.x && opts.batch.y) {
          const bx = opts.batch.x;
          const by = opts.batch.y;
          if (bx.length === by.length && bx.length > 0) batch = { x: bx, y: by };
        }
        if (!batch) batch = sampleBatch(V, batchSize, seed ^ (e * 2654435761));
        const bsz = batch.x.length;

        E.grad = null;
        const h = embeddingLookup(E, batch.x);
        const hn = rmsNorm(h, 1e-6);
        const logits = hn.matmul(E.transpose2d());
        const loss = crossEntropy(logits, batch.y);
        loss.backward();

        const g = E.grad || new Float32Array(embBlock);
        for (let i = 0; i < embBlock; i++) E.data[i] -= lr * g[i];
        E.markCPUDirty();

        lastLoss = loss.data[0];
        samples += bsz;
      }

      const out = new Float32Array(weights.length);
      out.set(weights);
      out.set(E.data, 0);
      return { updated: out, loss: Number(lastLoss), sampleCount: samples, mode: "v0-bbt-embed" };
    }

    const block = V * V;
    if (weights.length < block) {
      return { updated: weights, loss: 0.0, sampleCount: 0, mode: "v0-skipped" };
    }
    const wView = weights.subarray(0, block);
    const W = Tensor.fromArray(wView, [V, V], true);
    for (let e = 0; e < epochs; e++) {
      let batch = null;
      if (opts.batch && opts.batch.x && opts.batch.y) {
        const bx = opts.batch.x;
        const by = opts.batch.y;
        if (bx.length === by.length && bx.length > 0) batch = { x: bx, y: by };
      }
      if (!batch) batch = sampleBatch(V, batchSize, seed ^ (e * 2654435761));
      const bsz = batch.x.length;
      const X = Tensor.zeros([bsz, V], false);
      for (let i = 0; i < bsz; i++) X.data[i * V + batch.x[i]] = 1.0;
      W.grad = null;
      const logits = X.matmul(W);
      const loss = crossEntropy(logits, batch.y);
      loss.backward();
      const g = W.grad || new Float32Array(block);
      for (let i = 0; i < block; i++) W.data[i] -= lr * g[i];
      W.markCPUDirty();
      lastLoss = loss.data[0];
      samples += bsz;
    }
    const out = new Float32Array(weights.length);
    out.set(weights);
    out.set(W.data, 0);
    return { updated: out, loss: Number(lastLoss), sampleCount: samples, mode: "v0-autograd-bigram" };
  }

  async function debugAssertGPUHead(tensor, label = "tensor", count = 8, tol = 1e-4) {
    const backend = gpu();
    if (!backend || !tensor || !tensor.gpuBuffer) return true;
    const n = Math.max(1, Math.min(count, tensor.numel));
    const gpuHead = await backend.readBuffer(tensor.gpuBuffer, n);
    for (let i = 0; i < n; i++) {
      const a = tensor.data[i];
      const b = gpuHead[i];
      if (Math.abs(a - b) > tol) {
        throw new Error(`${label} gpu-sync mismatch at ${i}: cpu=${a} gpu=${b}`);
      }
    }
    return true;
  }

  async function trainLocalV0Async(weights, opts) {
    const backend = gpu();
    if (!backend) return trainLocalV0(weights, opts);

    const V = opts.vocabSize || 256;
    const D = opts.dModel || 64;
    const embBlock = V * D;

    if (weights.length < embBlock) return trainLocalV0(weights, opts);

    const lr = Math.max(1e-7, opts.lr || 5e-4);
    const epochs = Math.max(1, opts.epochs || 1);
    const batchSize = Math.max(8, opts.batchSize || 64);
    let seed = opts.seed >>> 0;

    let lastLoss = 0.0;
    let samples = 0;

    const eView = weights.subarray(0, embBlock);
    const E = Tensor.fromArray(eView, [V, D], true);
    E.toGPU();
    if (opts.debugSync) await debugAssertGPUHead(E, "E:init");

    for (let e = 0; e < epochs; e++) {
      let batch = null;
      if (opts.batch && opts.batch.x && opts.batch.y) {
        const bx = opts.batch.x;
        const by = opts.batch.y;
        if (bx.length === by.length && bx.length > 0) batch = { x: bx, y: by };
      }
      if (!batch) batch = sampleBatch(V, batchSize, seed ^ (e * 2654435761));
      const bsz = batch.x.length;

      E.grad = null;
      const h = embeddingLookup(E, batch.x);
      const hn = rmsNorm(h, 1e-6);
      const Et = E.transpose2d();
      const logits = hn.matmul(Et);

      if (logits.onGPU) await logits.toCPU();
      const loss = crossEntropy(logits, batch.y);
      loss.backward();
      await loss.resolveGrads();

      const g = E.grad || new Float32Array(embBlock);
      for (let i = 0; i < embBlock; i++) E.data[i] -= lr * g[i];
      E.markCPUDirty();

      if (e < epochs - 1) {
        E.toGPU();
        if (opts.debugSync) await debugAssertGPUHead(E, `E:epoch${e + 1}`);
      }

      lastLoss = loss.data[0];
      samples += bsz;
    }

    E.releaseGPU();
    const out = new Float32Array(weights.length);
    out.set(weights);
    out.set(E.data, 0);
    return { updated: out, loss: Number(lastLoss), sampleCount: samples, mode: "v0-bbt-embed-webgpu" };
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
    const expectedSize = model.paramCount();
    if (weights.length < expectedSize) {
      // Weights don't match full transformer — fall back to v0
      return trainLocalV0(weights, opts);
    }

    const hasGPU = !!(window.WebGPUBackend && WebGPUBackend.instance);
    model.loadFlatWeights(weights, hasGPU);
    const params = model.parameters();

    const lr = Math.max(1e-7, opts.lr || 5e-4);
    const epochs = Math.max(1, opts.epochs || 1);
    const batchSize = Math.max(2, opts.batchSize || 8);
    let seed = opts.seed >>> 0;

    let lastLoss = 0.0;
    let samples = 0;
    const backend = hasGPU ? WebGPUBackend.instance : null;

    // Create optimizer (handles GPU/CPU Adam/SGD state internally)
    const opt = new Optimizer(params, {
      lr, optimizer: opts.optimizer || "adam",
      beta1: opts.adamBeta1 || 0.9, beta2: opts.adamBeta2 || 0.999,
      gradClipNorm: opts.gradClipNorm != null ? opts.gradClipNorm : 1.0,
      warmupSteps: opts.warmupSteps || 10,
    });

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
      if (e + 1 < epochs) nextData = prepEpochData(e + 1);

      // Forward + loss (auto-dispatches to GPU or CPU)
      const logits = await model.forward(xFlat, actualB, T, ternarize);
      const loss = await Tensor.crossEntropy(logits, yFlat);

      // Backward + optimizer step (handles GPU/CPU internally)
      loss.backward();
      await opt.resolveAndClip(loss);
      await opt.step();

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
    if (hasGPU) {
      await model.toCPU();
    }

    // Release optimizer state + GPU buffers
    opt.release();
    if (hasGPU) {
      model.releaseGPU();
    }

    // Re-serialize
    const updated = model.toFlatWeights();

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

  // Attach to global
  window.tinygradV0.BBTTransformer = BBTTransformer;
  window.tinygradV0.trainLocalV0 = trainLocalV0;
  window.tinygradV0.trainLocalV0Async = trainLocalV0Async;
  window.tinygradV0.trainLocalV1 = trainLocalV1;
  window.tinygradV0.sampleBatch = sampleBatch;
  window.tinygradV0.debugAssertGPUHead = debugAssertGPUHead;
})();
