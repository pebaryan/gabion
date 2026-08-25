// tinygrad_v0.js
// Minimal browser-side autograd runtime for federated local training prototype.
// Supports optional WebGPU acceleration via WebGPUBackend (GPU path with CPU fallback).
(function () {
  "use strict";

  // --- GPU helper: check if backend is available ---
  function gpu() {
    return window.WebGPUBackend && WebGPUBackend.instance;
  }

  function getTraining() {
    return !!(window.tinygradV0 && window.tinygradV0.training);
  }

  function _matmul2d(A, ar, ac, B, br, bc) {
    const C = new Float32Array(ar * bc);
    for (let i = 0; i < ar; i++) {
      for (let k = 0; k < ac; k++) {
        const aik = A[i * ac + k];
        if (aik === 0) continue;
        const brow = k * bc;
        const crow = i * bc;
        for (let j = 0; j < bc; j++) C[crow + j] += aik * B[brow + j];
      }
    }
    return C;
  }

  function _transpose2d(A, r, c) {
    const T = new Float32Array(r * c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j * r + i] = A[i * c + j];
    return T;
  }

  /** Newton–Schulz odd polynomial, matching tinygrad Tensor.newton_schulz. */
  function newtonSchulz(data, rows, cols, steps, params, eps = 1e-7) {
    let G = data;
    let r = rows, c = cols;
    let transposed = false;
    if (r > c) {
      G = _transpose2d(G, r, c);
      const tmp = r; r = c; c = tmp;
      transposed = true;
    }
    let nrm = 0;
    for (let i = 0; i < G.length; i++) nrm += G[i] * G[i];
    nrm = Math.sqrt(nrm) + eps;
    const Gn = new Float32Array(G.length);
    for (let i = 0; i < G.length; i++) Gn[i] = G[i] / nrm;
    G = Gn;
    for (let s = 0; s < steps; s++) {
      const GT = _transpose2d(G, r, c);
      const GGT = _matmul2d(G, r, c, GT, c, r);
      const acc = new Float32Array(G.length);
      let X = G;
      for (let i = 0; i < params.length; i++) {
        if (i > 0) X = _matmul2d(GGT, r, r, X, r, c);
        const p = params[i];
        for (let k = 0; k < acc.length; k++) acc[k] += p * X[k];
      }
      G = acc;
    }
    return transposed ? _transpose2d(G, r, c) : G;
  }

  class Tensor {
    /**
     * @param {Float32Array} data - CPU data
     * @param {number[]} shape
     * @param {boolean} requiresGrad
     * @param {Tensor[]} parents
     * @param {function} backward
     * @param {GPUBuffer|null} gpuBuffer - optional GPU storage
     * @param {string} _dirty - 'cpu' | 'gpu' — which side is authoritative
     */
    constructor(data, shape, requiresGrad = false, parents = [], backward = null, gpuBuffer = null, _dirty = "cpu") {
      this.data = data;
      this.shape = shape;
      this.requiresGrad = !!requiresGrad;
      this.grad = null;
      this._parents = parents;
      this._backward = backward || (() => {});
      this.gpuBuffer = gpuBuffer;
      this._dirty = _dirty;
      this._isView = false; // true if gpuBuffer is shared (don't destroy on releaseGPU)
    }

    get numel() {
      return this.shape.reduce((a, b) => a * b, 1);
    }

    static zeros(shape, requiresGrad = false) {
      const n = shape.reduce((a, b) => a * b, 1);
      return new Tensor(new Float32Array(n), shape, requiresGrad);
    }

    static fromArray(arr, shape, requiresGrad = false) {
      const f = new Float32Array(arr);
      return new Tensor(f, shape, requiresGrad);
    }

    /** Upload CPU data to GPU buffer. Returns this for chaining. */
    toGPU() {
      const backend = gpu();
      if (!backend) return this;
      if (this.gpuBuffer && this._dirty === "gpu") return this;
      if (this.gpuBuffer && this._dirty === "cpu") {
        backend.writeBuffer(this.gpuBuffer, this.data);
      } else {
        this.gpuBuffer = backend.createBufferFromData(this.data);
      }
      this._dirty = "gpu";
      return this;
    }

    /** Async readback: GPU -> CPU. Updates this.data in place. */
    async toCPU() {
      if (this._dirty !== "gpu" || !this.gpuBuffer) return this;
      const backend = gpu();
      if (!backend) return this;
      this.data = await backend.readBuffer(this.gpuBuffer, this.numel);
      this._dirty = "cpu";
      return this;
    }

    /** Ensure data is on GPU (upload if needed). */
    ensureGPU() {
      if (!gpu()) return false;
      if (!this.gpuBuffer || this._dirty === "cpu") this.toGPU();
      return true;
    }

    /** Synchronous check: can we use GPU path for this tensor? */
    get onGPU() {
      return !!this.gpuBuffer && this._dirty === "gpu";
    }

    /** Release GPU buffer. Views (shared buffers) are detached without destroying. */
    releaseGPU() {
      if (this.gpuBuffer) {
        if (!this._isView) this.gpuBuffer.destroy();
        this.gpuBuffer = null;
      }
      this._isView = false;
      this._dirty = "cpu";
    }

    /** Mark CPU-side data as authoritative after in-place edits to this.data. */
    markCPUDirty() {
      this._dirty = "cpu";
      return this;
    }

    matmul(other) {
      const [m, k1] = this.shape;
      const [k2, n] = other.shape;
      if (k1 !== k2) throw new Error(`matmul shape mismatch: ${this.shape} x ${other.shape}`);

      const backend = gpu();
      if (backend) {
        // Auto-upload: if either side is on GPU, upload the other
        if (this.onGPU && !other.onGPU) other.toGPU();
        else if (!this.onGPU && other.onGPU) this.toGPU();

        if (this.onGPU && other.onGPU) {
          return this._matmulGPU(other, backend, m, k1, n);
        }
      }
      return this._matmulCPU(other, m, k1, n);
    }

    _matmulGPU(other, backend, m, k, n) {
      const outBuf = backend.matmul(this.gpuBuffer, other.gpuBuffer, m, k, n);
      const req = this.requiresGrad || other.requiresGrad;
      // CPU data placeholder (stale until readback)
      const outData = new Float32Array(m * n);
      return new Tensor(
        outData,
        [m, n],
        req,
        [this, other],
        (gout, goutBuf) => {
          // Backward for GPU matmul
          // dA = gout @ B^T  (gout[M,N] @ B^T[N,K] = [M,K])
          if (this.requiresGrad) {
            if (!this.grad) this.grad = new Float32Array(this.data.length);
            if (goutBuf && other.gpuBuffer && backend) {
              // GPU backward: dA = gout @ B^T
              const daBuf = backend.matmul(goutBuf, other.gpuBuffer, m, n, k, true);
              // Accumulate into grad on CPU (async handled by caller)
              this._gradGPUBuf = this._gradGPUBuf || null;
              this._pendingGradBuf = daBuf;
            } else {
              // CPU fallback
              const a = this.data;
              const b = other.data;
              for (let i = 0; i < m; i++) {
                for (let kk = 0; kk < k; kk++) {
                  let s = 0.0;
                  for (let j = 0; j < n; j++) s += gout[i * n + j] * b[kk * n + j];
                  this.grad[i * k + kk] += s;
                }
              }
            }
          }
          // dB = A^T @ gout  (A^T[K,M] @ gout[M,N] = [K,N])
          if (other.requiresGrad) {
            if (!other.grad) other.grad = new Float32Array(other.data.length);
            if (goutBuf && this.gpuBuffer && backend) {
              const dbBuf = backend.matmul(this.gpuBuffer, goutBuf, k, m, n, true);
              other._pendingGradBuf = dbBuf;
            } else {
              const a = this.data;
              const b = other.data;
              for (let kk = 0; kk < k; kk++) {
                for (let j = 0; j < n; j++) {
                  let s = 0.0;
                  for (let i = 0; i < m; i++) s += a[i * k + kk] * gout[i * n + j];
                  other.grad[kk * n + j] += s;
                }
              }
            }
          }
        },
        outBuf,
        "gpu"
      );
    }

    _matmulCPU(other, m, k1, n) {
      const out = new Float32Array(m * n);
      const a = this.data;
      const b = other.data;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0.0;
          for (let k = 0; k < k1; k++) s += a[i * k1 + k] * b[k * n + j];
          out[i * n + j] = s;
        }
      }

      const req = this.requiresGrad || other.requiresGrad;
      return new Tensor(
        out,
        [m, n],
        req,
        [this, other],
        (gout) => {
          if (this.requiresGrad) {
            if (!this.grad) this.grad = new Float32Array(this.data.length);
            for (let i = 0; i < m; i++) {
              for (let k = 0; k < k1; k++) {
                let s = 0.0;
                for (let j = 0; j < n; j++) s += gout[i * n + j] * b[k * n + j];
                this.grad[i * k1 + k] += s;
              }
            }
          }
          if (other.requiresGrad) {
            if (!other.grad) other.grad = new Float32Array(other.data.length);
            for (let k = 0; k < k1; k++) {
              for (let j = 0; j < n; j++) {
                let s = 0.0;
                for (let i = 0; i < m; i++) s += a[i * k1 + k] * gout[i * n + j];
                other.grad[k * n + j] += s;
              }
            }
          }
        }
      );
    }

    // --- Elementwise ops with autograd ---

    /** Elementwise add. Shapes must match. */
    add(other) {
      const n = this.numel;
      if (other.numel !== n) throw new Error(`add shape mismatch: ${this.shape} vs ${other.shape}`);
      const req = this.requiresGrad || other.requiresGrad;

      const backend = gpu();
      if (backend) {
        if (this.onGPU && !other.onGPU) other.toGPU();
        else if (!this.onGPU && other.onGPU) this.toGPU();

        if (this.onGPU && other.onGPU) {
          const outBuf = backend.elementwise(this.gpuBuffer, other.gpuBuffer, n, 0);
          const t = new Tensor(new Float32Array(n), [...this.shape], req, [this, other], (gout, goutBuf) => {
            // Gradient of add passes through unchanged to both parents
            if (goutBuf) {
              if (this.requiresGrad) this._pendingGradBuf = goutBuf;
              if (other.requiresGrad) other._pendingGradBuf = goutBuf;
            } else {
              if (this.requiresGrad) {
                if (!this.grad) this.grad = new Float32Array(n);
                for (let i = 0; i < n; i++) this.grad[i] += gout[i];
              }
              if (other.requiresGrad) {
                if (!other.grad) other.grad = new Float32Array(n);
                for (let i = 0; i < n; i++) other.grad[i] += gout[i];
              }
            }
          });
          t.gpuBuffer = outBuf;
          t._dirty = "gpu";
          return t;
        }
      }

      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = this.data[i] + other.data[i];
      return new Tensor(out, [...this.shape], req, [this, other], (gout) => {
        if (this.requiresGrad) {
          if (!this.grad) this.grad = new Float32Array(n);
          for (let i = 0; i < n; i++) this.grad[i] += gout[i];
        }
        if (other.requiresGrad) {
          if (!other.grad) other.grad = new Float32Array(n);
          for (let i = 0; i < n; i++) other.grad[i] += gout[i];
        }
      });
    }

    /** Elementwise multiply. Shapes must match. */
    mul(other) {
      const n = this.numel;
      if (other.numel !== n) throw new Error(`mul shape mismatch: ${this.shape} vs ${other.shape}`);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = this.data[i] * other.data[i];
      const req = this.requiresGrad || other.requiresGrad;
      const aData = this.data;
      const bData = other.data;
      // Capture GPU buffers for backward
      const aBuf = this.gpuBuffer;
      const bBuf = other.gpuBuffer;
      const aOnGPU = this.onGPU;
      const bOnGPU = other.onGPU;
      return new Tensor(out, [...this.shape], req, [this, other], (gout) => {
        const backend = gpu();
        if (backend && (aOnGPU || bOnGPU)) {
          const goutBuf = backend.createBufferFromData(gout);
          // dA = gout * B → elementwise mul (op=1)
          if (this.requiresGrad && bBuf) {
            this._pendingGradBuf = backend.elementwise(goutBuf, bBuf, n, 1);
          } else if (this.requiresGrad) {
            if (!this.grad) this.grad = new Float32Array(n);
            for (let i = 0; i < n; i++) this.grad[i] += gout[i] * bData[i];
          }
          // dB = gout * A → elementwise mul (op=1)
          if (other.requiresGrad && aBuf) {
            other._pendingGradBuf = backend.elementwise(goutBuf, aBuf, n, 1);
          } else if (other.requiresGrad) {
            if (!other.grad) other.grad = new Float32Array(n);
            for (let i = 0; i < n; i++) other.grad[i] += gout[i] * aData[i];
          }
          backend.releaseBuffer(goutBuf);
        } else {
          if (this.requiresGrad) {
            if (!this.grad) this.grad = new Float32Array(n);
            for (let i = 0; i < n; i++) this.grad[i] += gout[i] * bData[i];
          }
          if (other.requiresGrad) {
            if (!other.grad) other.grad = new Float32Array(n);
            for (let i = 0; i < n; i++) other.grad[i] += gout[i] * aData[i];
          }
        }
      });
    }

    /** Negate. */
    neg() {
      const n = this.numel;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = -this.data[i];
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] -= gout[i];
      });
    }

    /** Detach: creates a new tensor with same data but no gradient tracking.
     *  Used for Straight-Through Estimator (STE). */
    detach() {
      return new Tensor(this.data, [...this.shape], false, [], null);
    }

    /** Clip (clamp) values to [lo, hi]. Backward: pass through where not clipped. */
    clip(lo, hi) {
      const n = this.numel;
      const out = new Float32Array(n);
      const mask = new Uint8Array(n); // 1 = not clipped
      for (let i = 0; i < n; i++) {
        const v = this.data[i];
        if (v < lo) { out[i] = lo; mask[i] = 0; }
        else if (v > hi) { out[i] = hi; mask[i] = 0; }
        else { out[i] = v; mask[i] = 1; }
      }
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i] * mask[i];
      });
    }

    /** Round to nearest integer. Backward: STE (pass through). */
    round() {
      const n = this.numel;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.round(this.data[i]);
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i]; // STE: identity
      });
    }

    /** Elementwise subtract. Shapes must match. */
    sub(other) {
      return this.add(other.neg());
    }

    /** ReLU. */
    relu() {
      const n = this.numel;
      const out = new Float32Array(n);
      const mask = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const v = this.data[i];
        const keep = v > 0;
        out[i] = keep ? v : 0;
        mask[i] = keep ? 1 : 0;
      }
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i] * mask[i];
      });
    }

    /** GELU tanh approximation (tinygrad default). */
    gelu() {
      const n = this.numel;
      const out = new Float32Array(n);
      const dact = new Float32Array(n);
      const c = Math.sqrt(2 / Math.PI);
      for (let i = 0; i < n; i++) {
        const x = this.data[i];
        const u = c * (x + 0.044715 * x * x * x);
        const th = Math.tanh(u);
        out[i] = 0.5 * x * (1 + th);
        const sech2 = 1 - th * th;
        const du = c * (1 + 3 * 0.044715 * x * x);
        dact[i] = 0.5 * (1 + th) + 0.5 * x * sech2 * du;
      }
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i] * dact[i];
      });
    }

    /**
     * Last-axis layer norm (no affine). Matches tinygrad Tensor.layernorm(axis=-1).
     * Input [..., C] -> [..., C].
     */
    layerNorm(eps = 1e-5) {
      const shape = this.shape;
      const C = shape[shape.length - 1];
      const rows = this.numel / C;
      const backend = gpu();
      if (backend && this.onGPU) {
        const outBuf = backend.layernorm(this.gpuBuffer, rows, C, eps);
        const y = new Float32Array(this.numel);
        const t = new Tensor(y, [...shape], this.requiresGrad, [this], (gout) => {
          if (!this.requiresGrad) return;
          if (!this.grad) this.grad = new Float32Array(this.numel);
          for (let r = 0; r < rows; r++) {
            const off = r * C;
            let mean = 0, varr = 0;
            for (let j = 0; j < C; j++) mean += this.data[off + j];
            mean /= C;
            for (let j = 0; j < C; j++) { const d = this.data[off + j] - mean; varr += d * d; }
            const invStd = 1 / Math.sqrt(varr / C + eps);
            let sumG = 0, dot = 0;
            const yrow = new Float32Array(C);
            for (let j = 0; j < C; j++) { yrow[j] = (this.data[off + j] - mean) * invStd; sumG += gout[off + j]; dot += gout[off + j] * yrow[j]; }
            const meanG = sumG / C, meanGY = dot / C;
            for (let j = 0; j < C; j++) this.grad[off + j] += invStd * (gout[off + j] - meanG - yrow[j] * meanGY);
          }
        });
        t.gpuBuffer = outBuf;
        t._dirty = "gpu";
        return t;
      }
      const y = new Float32Array(this.numel);
      const mu = new Float32Array(rows);
      const inv = new Float32Array(rows);
      const xData = this.data;
      for (let r = 0; r < rows; r++) {
        const off = r * C;
        let s = 0;
        for (let j = 0; j < C; j++) s += xData[off + j];
        const m = s / C;
        mu[r] = m;
        let v = 0;
        for (let j = 0; j < C; j++) {
          const d = xData[off + j] - m;
          v += d * d;
        }
        const invStd = 1 / Math.sqrt(v / C + eps);
        inv[r] = invStd;
        for (let j = 0; j < C; j++) y[off + j] = (xData[off + j] - m) * invStd;
      }
      return new Tensor(y, [...shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(this.numel);
        for (let r = 0; r < rows; r++) {
          const off = r * C;
          const invStd = inv[r];
          let dot = 0;
          let sumG = 0;
          for (let j = 0; j < C; j++) {
            sumG += gout[off + j];
            dot += gout[off + j] * y[off + j];
          }
          const meanG = sumG / C;
          const meanGY = dot / C;
          for (let j = 0; j < C; j++) {
            this.grad[off + j] += invStd * (gout[off + j] - meanG - y[off + j] * meanGY);
          }
        }
      });
    }

    /**
     * Affine on last dim: y = x * weight[+bias]. weight/bias are 1D length C.
     */
    affineLast(weight, bias = null) {
      const C = this.shape[this.shape.length - 1];
      if (weight.numel !== C) throw new Error(`affineLast weight ${weight.shape} != last dim ${C}`);
      if (bias && bias.numel !== C) throw new Error(`affineLast bias ${bias.shape} != last dim ${C}`);
      const n = this.numel;
      const rows = n / C;
      const out = new Float32Array(n);
      const w = weight.data;
      const b = bias ? bias.data : null;
      const xData = this.data;
      for (let r = 0; r < rows; r++) {
        const off = r * C;
        for (let j = 0; j < C; j++) out[off + j] = xData[off + j] * w[j] + (b ? b[j] : 0);
      }
      const req = this.requiresGrad || weight.requiresGrad || !!(bias && bias.requiresGrad);
      const parents = bias ? [this, weight, bias] : [this, weight];
      return new Tensor(out, [...this.shape], req, parents, (gout) => {
        if (this.requiresGrad) {
          if (!this.grad) this.grad = new Float32Array(n);
          for (let r = 0; r < rows; r++) {
            const off = r * C;
            for (let j = 0; j < C; j++) this.grad[off + j] += gout[off + j] * w[j];
          }
        }
        if (weight.requiresGrad) {
          if (!weight.grad) weight.grad = new Float32Array(C);
          for (let r = 0; r < rows; r++) {
            const off = r * C;
            for (let j = 0; j < C; j++) weight.grad[j] += gout[off + j] * xData[off + j];
          }
        }
        if (bias && bias.requiresGrad) {
          if (!bias.grad) bias.grad = new Float32Array(C);
          for (let r = 0; r < rows; r++) {
            const off = r * C;
            for (let j = 0; j < C; j++) bias.grad[j] += gout[off + j];
          }
        }
      });
    }

    /** Dropout. Active only when tinygradV0.training is true. */
    dropout(p = 0.5) {
      if (!(p >= 0 && p <= 1)) throw new Error(`dropout p=${p} out of range`);
      if (!getTraining() || p === 0) return this;
      if (p === 1) return Tensor.zeros(this.shape, this.requiresGrad);
      const backend = gpu();
      if (backend && this.onGPU) {
        const n = this.numel;
        const seed = (Math.random() * 0xffffffff) >>> 0;
        const outBuf = backend.dropoutFwd(this.gpuBuffer, n, p, seed);
        const out = new Float32Array(n);
        const t = new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
          if (!this.requiresGrad) return;
          if (!this.grad) this.grad = new Float32Array(n);
          const goutBuf = backend.createBufferFromData(gout);
          const dxBuf = backend.dropoutBwd(goutBuf, n, p, seed);
          this._pendingGradBuf = dxBuf;
          goutBuf.destroy();
        });
        t.gpuBuffer = outBuf;
        t._dirty = "gpu";
        t._dropoutSeed = seed;
        return t;
      }
      const n = this.numel;
      const out = new Float32Array(n);
      const mask = new Float32Array(n);
      const scale = 1 / (1 - p);
      for (let i = 0; i < n; i++) {
        const keep = Math.random() >= p;
        mask[i] = keep ? scale : 0;
        out[i] = this.data[i] * mask[i];
      }
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i] * mask[i];
      });
    }

    /** Add a 1D bias [C] onto [..., C]. */
    addBroadcastLast(bias) {
      const C = this.shape[this.shape.length - 1];
      if (bias.numel !== C) throw new Error("addBroadcastLast bias mismatch");
      const n = this.numel;
      const rows = n / C;
      const out = new Float32Array(n);
      for (let r = 0; r < rows; r++) {
        const off = r * C;
        for (let j = 0; j < C; j++) out[off + j] = this.data[off + j] + bias.data[j];
      }
      const req = this.requiresGrad || bias.requiresGrad;
      return new Tensor(out, [...this.shape], req, [this, bias], (gout) => {
        if (this.requiresGrad) {
          if (!this.grad) this.grad = new Float32Array(n);
          for (let i = 0; i < n; i++) this.grad[i] += gout[i];
        }
        if (bias.requiresGrad) {
          if (!bias.grad) bias.grad = new Float32Array(C);
          for (let r = 0; r < rows; r++) {
            const off = r * C;
            for (let j = 0; j < C; j++) bias.grad[j] += gout[off + j];
          }
        }
      });
    }

    /** Sigmoid. */
    sigmoid() {
      const n = this.numel;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-this.data[i]));
      const s = out;
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i] * s[i] * (1 - s[i]);
      });
    }

    /** Tanh. */
    tanh() {
      const n = this.numel;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.tanh(this.data[i]);
      const y = out;
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i] * (1 - y[i] * y[i]);
      });
    }

    /**
     * NCHW conv2d. weight [Cout, Cin/groups, kH, kW], bias [Cout] optional.
     * padding/stride/dilation: number or [h,w].
     */
    conv2d(weight, bias = null, groups = 1, stride = 1, dilation = 1, padding = 0) {
      if (this.shape.length !== 4) throw new Error(`conv2d input must be NCHW, got ${this.shape}`);
      if (weight.shape.length !== 4) throw new Error(`conv2d weight must be OIHW, got ${weight.shape}`);
      const [N, Cin, H, W] = this.shape;
      const [Cout, CinG, kH, kW] = weight.shape;
      if (Cin % groups !== 0 || Cout % groups !== 0) throw new Error("conv2d groups must divide Cin and Cout");
      if (CinG !== Cin / groups) throw new Error(`conv2d Cin/groups mismatch ${Cin}/${groups} vs ${CinG}`);
      const sH = Array.isArray(stride) ? stride[0] : stride;
      const sW = Array.isArray(stride) ? stride[1] : stride;
      const dH = Array.isArray(dilation) ? dilation[0] : dilation;
      const dW = Array.isArray(dilation) ? dilation[1] : dilation;
      const pH = Array.isArray(padding) ? padding[0] : padding;
      const pW = Array.isArray(padding) ? padding[1] : padding;
      const Ho = Math.floor((H + 2 * pH - dH * (kH - 1) - 1) / sH) + 1;
      const Wo = Math.floor((W + 2 * pW - dW * (kW - 1) - 1) / sW) + 1;
      if (Ho <= 0 || Wo <= 0) throw new Error(`conv2d empty output H=${H} W=${W} k=${kH}x${kW}`);
      const cinPerG = Cin / groups;
      const coutPerG = Cout / groups;
      const x = this.data, w = weight.data, b = bias ? bias.data : null;
      const out = new Float32Array(N * Cout * Ho * Wo);
      const xStr = [Cin * H * W, H * W, W, 1];
      const wStr = [cinPerG * kH * kW, kH * kW, kW, 1];
      const oStr = [Cout * Ho * Wo, Ho * Wo, Wo, 1];
      for (let n = 0; n < N; n++) {
        for (let oc = 0; oc < Cout; oc++) {
          const g = Math.floor(oc / coutPerG);
          const ic0 = g * cinPerG;
          for (let oh = 0; oh < Ho; oh++) {
            for (let ow = 0; ow < Wo; ow++) {
              let acc = b ? b[oc] : 0;
              const ih0 = oh * sH - pH;
              const iw0 = ow * sW - pW;
              for (let icL = 0; icL < cinPerG; icL++) {
                const ic = ic0 + icL;
                for (let kh = 0; kh < kH; kh++) {
                  const ih = ih0 + kh * dH;
                  if (ih < 0 || ih >= H) continue;
                  for (let kw = 0; kw < kW; kw++) {
                    const iw = iw0 + kw * dW;
                    if (iw < 0 || iw >= W) continue;
                    acc += x[n * xStr[0] + ic * xStr[1] + ih * xStr[2] + iw] *
                           w[oc * wStr[0] + icL * wStr[1] + kh * wStr[2] + kw];
                  }
                }
              }
              out[n * oStr[0] + oc * oStr[1] + oh * oStr[2] + ow] = acc;
            }
          }
        }
      }
      const req = this.requiresGrad || weight.requiresGrad || !!(bias && bias.requiresGrad);
      const parents = bias ? [this, weight, bias] : [this, weight];
      const P = { N, Cin, H, W, Cout, Ho, Wo, kH, kW, groups, cinPerG, coutPerG, sH, sW, dH, dW, pH, pW, hasBias: bias ? 1 : 0 };
      const cpuBackward = (gout) => {
        if (this.requiresGrad) {
          if (!this.grad) this.grad = new Float32Array(this.numel);
          const dx = this.grad;
          for (let n = 0; n < N; n++) for (let oc = 0; oc < Cout; oc++) {
            const g = Math.floor(oc / coutPerG), ic0 = g * cinPerG;
            for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
              const gv = gout[n * oStr[0] + oc * oStr[1] + oh * oStr[2] + ow];
              const ih0 = oh * sH - pH, iw0 = ow * sW - pW;
              for (let icL = 0; icL < cinPerG; icL++) {
                const ic = ic0 + icL;
                for (let kh = 0; kh < kH; kh++) {
                  const ih = ih0 + kh * dH; if (ih < 0 || ih >= H) continue;
                  for (let kw = 0; kw < kW; kw++) {
                    const iw = iw0 + kw * dW; if (iw < 0 || iw >= W) continue;
                    dx[n * xStr[0] + ic * xStr[1] + ih * xStr[2] + iw] +=
                      gv * w[oc * wStr[0] + icL * wStr[1] + kh * wStr[2] + kw];
                  }
                }
              }
            }
          }
        }
        if (weight.requiresGrad) {
          if (!weight.grad) weight.grad = new Float32Array(weight.numel);
          const dw = weight.grad;
          for (let n = 0; n < N; n++) for (let oc = 0; oc < Cout; oc++) {
            const g = Math.floor(oc / coutPerG), ic0 = g * cinPerG;
            for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++) {
              const gv = gout[n * oStr[0] + oc * oStr[1] + oh * oStr[2] + ow];
              const ih0 = oh * sH - pH, iw0 = ow * sW - pW;
              for (let icL = 0; icL < cinPerG; icL++) {
                const ic = ic0 + icL;
                for (let kh = 0; kh < kH; kh++) {
                  const ih = ih0 + kh * dH; if (ih < 0 || ih >= H) continue;
                  for (let kw = 0; kw < kW; kw++) {
                    const iw = iw0 + kw * dW; if (iw < 0 || iw >= W) continue;
                    dw[oc * wStr[0] + icL * wStr[1] + kh * wStr[2] + kw] +=
                      gv * x[n * xStr[0] + ic * xStr[1] + ih * xStr[2] + iw];
                  }
                }
              }
            }
          }
        }
        if (bias && bias.requiresGrad) {
          if (!bias.grad) bias.grad = new Float32Array(Cout);
          for (let n = 0; n < N; n++) for (let oc = 0; oc < Cout; oc++)
            for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++)
              bias.grad[oc] += gout[n * oStr[0] + oc * oStr[1] + oh * oStr[2] + ow];
        }
      };
      const backend = gpu();
      if (backend && this.onGPU && weight.onGPU && (!bias || bias.onGPU)) {
        const outBuf = backend.conv2d(this.gpuBuffer, weight.gpuBuffer, bias ? bias.gpuBuffer : null, P);
        return new Tensor(new Float32Array(N * Cout * Ho * Wo), [N, Cout, Ho, Wo], req, parents, (gout, goutBuf) => {
          if (goutBuf && backend) {
            if (this.requiresGrad) this._pendingGradBuf = backend.conv2dBwdDx(weight.gpuBuffer, goutBuf, P);
            if (weight.requiresGrad) weight._pendingGradBuf = backend.conv2dBwdDw(this.gpuBuffer, goutBuf, P);
            if (bias && bias.requiresGrad) bias._pendingGradBuf = backend.convBwdDb(goutBuf, P);
          } else {
            cpuBackward(gout);
          }
        }, outBuf, "gpu");
      }
      return new Tensor(out, [N, Cout, Ho, Wo], req, parents, cpuBackward);
    }

    /**
     * NCHW conv_transpose2d. weight [Cin, Cout/groups, kH, kW] (tinygrad ConvTranspose2d layout).
     */
    convTranspose2d(weight, bias = null, groups = 1, stride = 1, dilation = 1, padding = 0, outputPadding = 0) {
      if (this.shape.length !== 4) throw new Error(`conv_transpose2d input must be NCHW, got ${this.shape}`);
      if (weight.shape.length !== 4) throw new Error(`conv_transpose2d weight must be [Cin,Cout/g,kH,kW], got ${weight.shape}`);
      const [N, Cin, H, W] = this.shape;
      const [CinW, CoutG, kH, kW] = weight.shape;
      if (CinW !== Cin) throw new Error("conv_transpose2d Cin mismatch");
      if (Cin % groups !== 0) throw new Error("conv_transpose2d groups must divide Cin");
      const sH = Array.isArray(stride) ? stride[0] : stride;
      const sW = Array.isArray(stride) ? stride[1] : stride;
      const dH = Array.isArray(dilation) ? dilation[0] : dilation;
      const dW = Array.isArray(dilation) ? dilation[1] : dilation;
      const pH = Array.isArray(padding) ? padding[0] : padding;
      const pW = Array.isArray(padding) ? padding[1] : padding;
      const oH = Array.isArray(outputPadding) ? outputPadding[0] : outputPadding;
      const oW = Array.isArray(outputPadding) ? outputPadding[1] : outputPadding;
      const Cout = CoutG * groups;
      const cinPerG = Cin / groups;
      const Ho = (H - 1) * sH - 2 * pH + dH * (kH - 1) + oH + 1;
      const Wo = (W - 1) * sW - 2 * pW + dW * (kW - 1) + oW + 1;
      const x = this.data, wgt = weight.data, b = bias ? bias.data : null;
      const out = new Float32Array(N * Cout * Ho * Wo);
      if (b) {
        for (let n = 0; n < N; n++) for (let oc = 0; oc < Cout; oc++)
          for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++)
            out[((n * Cout + oc) * Ho + oh) * Wo + ow] = b[oc];
      }
      const xStr = [Cin * H * W, H * W, W, 1];
      const wStr = [CoutG * kH * kW, kH * kW, kW, 1];
      const oStr = [Cout * Ho * Wo, Ho * Wo, Wo, 1];
      for (let n = 0; n < N; n++) {
        for (let ic = 0; ic < Cin; ic++) {
          const g = Math.floor(ic / cinPerG);
          const oc0 = g * CoutG;
          for (let ih = 0; ih < H; ih++) for (let iw = 0; iw < W; iw++) {
            const xv = x[n * xStr[0] + ic * xStr[1] + ih * xStr[2] + iw];
            for (let ocL = 0; ocL < CoutG; ocL++) {
              const oc = oc0 + ocL;
              for (let kh = 0; kh < kH; kh++) {
                const oh = ih * sH - pH + kh * dH;
                if (oh < 0 || oh >= Ho) continue;
                for (let kw = 0; kw < kW; kw++) {
                  const ow = iw * sW - pW + kw * dW;
                  if (ow < 0 || ow >= Wo) continue;
                  // spatial index as-is (tinygrad flips weight then conv; we scatter without extra flip)
                  out[n * oStr[0] + oc * oStr[1] + oh * oStr[2] + ow] +=
                    xv * wgt[ic * wStr[0] + ocL * wStr[1] + kh * wStr[2] + kw];
                }
              }
            }
          }
        }
      }
      const req = this.requiresGrad || weight.requiresGrad || !!(bias && bias.requiresGrad);
      const parents = bias ? [this, weight, bias] : [this, weight];
      const P = { N, Cin, H, W, Cout, Ho, Wo, kH, kW, groups, cinPerG, coutPerG: CoutG, sH, sW, dH, dW, pH, pW, hasBias: bias ? 1 : 0 };
      const cpuBackward = (gout) => {
        // ConvTranspose2d backward: mirrors the forward scatter as a gather.
        // dx: each input element sums gout at every output position it wrote to.
        // dw: each weight element accumulates x * gout over the same (oh, ow) hits.
        // oh = ih*sH - pH + kh*dH, ow = iw*sW - pW + kw*dW (same as forward).
        if (this.requiresGrad) {
          if (!this.grad) this.grad = new Float32Array(this.numel);
          const dx = this.grad;
          for (let n = 0; n < N; n++) for (let ic = 0; ic < Cin; ic++) {
            const grp = Math.floor(ic / cinPerG);
            const oc0 = grp * CoutG;
            for (let ih = 0; ih < H; ih++) for (let iw = 0; iw < W; iw++) {
              let acc = 0;
              for (let ocL = 0; ocL < CoutG; ocL++) {
                const oc = oc0 + ocL;
                for (let kh = 0; kh < kH; kh++) {
                  const oh = ih * sH - pH + kh * dH;
                  if (oh < 0 || oh >= Ho) continue;
                  for (let kw = 0; kw < kW; kw++) {
                    const ow = iw * sW - pW + kw * dW;
                    if (ow < 0 || ow >= Wo) continue;
                    acc += gout[n * oStr[0] + oc * oStr[1] + oh * oStr[2] + ow] *
                           wgt[ic * wStr[0] + ocL * wStr[1] + kh * wStr[2] + kw];
                  }
                }
              }
              dx[n * xStr[0] + ic * xStr[1] + ih * xStr[2] + iw] += acc;
            }
          }
        }
        if (weight.requiresGrad) {
          if (!weight.grad) weight.grad = new Float32Array(weight.numel);
          const dw = weight.grad;
          for (let n = 0; n < N; n++) for (let ic = 0; ic < Cin; ic++) {
            const grp = Math.floor(ic / cinPerG);
            const oc0 = grp * CoutG;
            for (let ih = 0; ih < H; ih++) for (let iw = 0; iw < W; iw++) {
              const xv = x[n * xStr[0] + ic * xStr[1] + ih * xStr[2] + iw];
              for (let ocL = 0; ocL < CoutG; ocL++) {
                const oc = oc0 + ocL;
                for (let kh = 0; kh < kH; kh++) {
                  const oh = ih * sH - pH + kh * dH;
                  if (oh < 0 || oh >= Ho) continue;
                  for (let kw = 0; kw < kW; kw++) {
                    const ow = iw * sW - pW + kw * dW;
                    if (ow < 0 || ow >= Wo) continue;
                    dw[ic * wStr[0] + ocL * wStr[1] + kh * wStr[2] + kw] +=
                      xv * gout[n * oStr[0] + oc * oStr[1] + oh * oStr[2] + ow];
                  }
                }
              }
            }
          }
        }
        if (bias && bias.requiresGrad) {
          if (!bias.grad) bias.grad = new Float32Array(Cout);
          for (let n = 0; n < N; n++) for (let oc = 0; oc < Cout; oc++)
            for (let oh = 0; oh < Ho; oh++) for (let ow = 0; ow < Wo; ow++)
              bias.grad[oc] += gout[n * oStr[0] + oc * oStr[1] + oh * oStr[2] + ow];
        }
      };
      const backend = gpu();
      if (backend && this.onGPU && weight.onGPU && (!bias || bias.onGPU)) {
        const outBuf = backend.convTranspose2d(this.gpuBuffer, weight.gpuBuffer, bias ? bias.gpuBuffer : null, P);
        return new Tensor(new Float32Array(N * Cout * Ho * Wo), [N, Cout, Ho, Wo], req, parents, (gout, goutBuf) => {
          if (goutBuf && backend) {
            if (this.requiresGrad) this._pendingGradBuf = backend.convTranspose2dBwdDx(weight.gpuBuffer, goutBuf, P);
            if (weight.requiresGrad) weight._pendingGradBuf = backend.convTranspose2dBwdDw(this.gpuBuffer, goutBuf, P);
            if (bias && bias.requiresGrad) bias._pendingGradBuf = backend.convBwdDb(goutBuf, P);
          } else {
            cpuBackward(gout);
          }
        }, outBuf, "gpu");
      }
      return new Tensor(out, [N, Cout, Ho, Wo], req, parents, cpuBackward);
    }

    /** Affine on NCHW channel dim (axis=1). */
    affineChannel(weight, bias = null) {
      if (this.shape.length < 2) throw new Error("affineChannel needs rank>=2");
      const C = this.shape[1];
      if (weight.numel !== C) throw new Error("affineChannel weight mismatch");
      const N = this.shape[0];
      const rest = this.numel / (N * C);
      const out = new Float32Array(this.numel);
      const x = this.data, w = weight.data, b = bias ? bias.data : null;
      for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
        const off = (n * C + c) * rest;
        const wc = w[c], bc = b ? b[c] : 0;
        for (let i = 0; i < rest; i++) out[off + i] = x[off + i] * wc + bc;
      }
      const req = this.requiresGrad || weight.requiresGrad || !!(bias && bias.requiresGrad);
      const parents = bias ? [this, weight, bias] : [this, weight];
      const P = { N, C, rest, hasBias: bias ? 1 : 0 };
      const cpuBackward = (gout) => {
        if (this.requiresGrad) {
          if (!this.grad) this.grad = new Float32Array(this.numel);
          for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
            const off = (n * C + c) * rest;
            for (let i = 0; i < rest; i++) this.grad[off + i] += gout[off + i] * w[c];
          }
        }
        if (weight.requiresGrad) {
          if (!weight.grad) weight.grad = new Float32Array(C);
          for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
            const off = (n * C + c) * rest;
            let s = 0;
            for (let i = 0; i < rest; i++) s += gout[off + i] * x[off + i];
            weight.grad[c] += s;
          }
        }
        if (bias && bias.requiresGrad) {
          if (!bias.grad) bias.grad = new Float32Array(C);
          for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
            const off = (n * C + c) * rest;
            let s = 0;
            for (let i = 0; i < rest; i++) s += gout[off + i];
            bias.grad[c] += s;
          }
        }
      };
      const backend = gpu();
      if (backend && this.onGPU && weight.onGPU && (!bias || bias.onGPU)) {
        const outBuf = backend.affineChannel(this.gpuBuffer, weight.gpuBuffer, bias ? bias.gpuBuffer : null, P);
        return new Tensor(new Float32Array(this.numel), [...this.shape], req, parents, (gout, goutBuf) => {
          if (goutBuf && backend) {
            if (this.requiresGrad) this._pendingGradBuf = backend.elementwise(goutBuf, weight.gpuBuffer, this.numel, 1);
            if (weight.requiresGrad) weight._pendingGradBuf = backend.affineChannelBwdDw(this.gpuBuffer, goutBuf, P);
            if (bias && bias.requiresGrad) {
              // reuse conv_bwd_db: reduce gout over [N, C, rest] per channel (Ho=rest, Wo=1)
              const dbP = { N, Cin: 1, H: 1, W: 1, Cout: C, Ho: rest, Wo: 1, kH: 1, kW: 1, groups: 1, cinPerG: 1, coutPerG: C, sH: 1, sW: 1, dH: 1, dW: 1, pH: 0, pW: 0, hasBias: 1 };
              bias._pendingGradBuf = backend.convBwdDb(goutBuf, dbP);
            }
          } else {
            cpuBackward(gout);
          }
        }, outBuf, "gpu");
      }
      return new Tensor(out, [...this.shape], req, parents, cpuBackward);
    }

    /** Elementwise absolute value. Backward: sign(x). */
    abs() {
      const n = this.numel;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.abs(this.data[i]);
      const xData = this.data;
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          this.grad[i] += gout[i] * (xData[i] >= 0 ? 1 : -1);
        }
      });
    }

    /** Mean of all elements -> scalar tensor. */
    meanAll() {
      const n = this.numel;
      let s = 0.0;
      for (let i = 0; i < n; i++) s += this.data[i];
      const mean = s / n;
      return new Tensor(new Float32Array([mean]), [1], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        const g = (gout[0] || 1.0) / n;
        for (let i = 0; i < n; i++) this.grad[i] += g;
      });
    }

    /** Max along last axis with keepdim. Input [..., C] -> [..., 1]. No gradient (used for scaling). */
    maxLastAxis() {
      const shape = this.shape;
      const C = shape[shape.length - 1];
      const rows = this.numel / C;
      const out = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        let mx = -Infinity;
        const off = r * C;
        for (let j = 0; j < C; j++) {
          const v = this.data[off + j];
          if (v > mx) mx = v;
        }
        out[r] = mx;
      }
      // Return [rows, 1] with keepdim
      const outShape = [...shape.slice(0, -1), 1];
      return new Tensor(out, outShape, false, [], null);
    }

    /** Elementwise divide by another tensor (with broadcasting for last-dim=1).
     *  other must have same total rows but last dim = 1 (broadcast) or same shape. */
    div(other) {
      const n = this.numel;
      const shape = this.shape;
      const C = shape[shape.length - 1];
      const rows = n / C;
      const oC = other.shape[other.shape.length - 1];
      const out = new Float32Array(n);

      if (oC === 1 && other.numel === rows) {
        // Broadcasting: other is [rows, 1], divide each row
        for (let r = 0; r < rows; r++) {
          const d = other.data[r];
          const off = r * C;
          for (let j = 0; j < C; j++) out[off + j] = this.data[off + j] / d;
        }
      } else if (other.numel === n) {
        for (let i = 0; i < n; i++) out[i] = this.data[i] / other.data[i];
      } else {
        throw new Error(`div shape mismatch: ${this.shape} / ${other.shape}`);
      }

      const req = this.requiresGrad || other.requiresGrad;
      const aData = this.data;
      const bData = other.data;
      return new Tensor(out, [...shape], req, [this, other], (gout) => {
        if (this.requiresGrad) {
          if (!this.grad) this.grad = new Float32Array(n);
          if (oC === 1 && other.numel === rows) {
            for (let r = 0; r < rows; r++) {
              const d = bData[r];
              const off = r * C;
              for (let j = 0; j < C; j++) this.grad[off + j] += gout[off + j] / d;
            }
          } else {
            for (let i = 0; i < n; i++) this.grad[i] += gout[i] / bData[i];
          }
        }
        if (other.requiresGrad) {
          if (!other.grad) other.grad = new Float32Array(other.data.length);
          if (oC === 1 && other.numel === rows) {
            for (let r = 0; r < rows; r++) {
              const d = bData[r];
              const off = r * C;
              let s = 0.0;
              for (let j = 0; j < C; j++) s += gout[off + j] * aData[off + j];
              other.grad[r] += -s / (d * d);
            }
          } else {
            for (let i = 0; i < n; i++) other.grad[i] += -gout[i] * aData[i] / (bData[i] * bData[i]);
          }
        }
      });
    }

    /** Scale by scalar. Auto-dispatches to GPU. */
    scale(s) {
      const n = this.numel;
      const backend = gpu();
      if (backend && this.onGPU) {
        const outBuf = backend.elementwise(this.gpuBuffer, this.gpuBuffer, n, 3, s);
        const out = new Tensor(new Float32Array(n), [...this.shape], this.requiresGrad, [this], (gout, goutBuf) => {
          if (!this.requiresGrad) return;
          if (goutBuf && backend) {
            this._pendingGradBuf = backend.elementwise(goutBuf, goutBuf, n, 3, s);
          } else {
            if (!this.grad) this.grad = new Float32Array(n);
            for (let i = 0; i < n; i++) this.grad[i] += gout[i] * s;
          }
        });
        out.gpuBuffer = outBuf;
        out._dirty = "gpu";
        return out;
      }
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = this.data[i] * s;
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i] * s;
      });
    }

    /** Reshape (view, no copy). Product of dims must match numel. Propagates GPU buffer. */
    reshape(newShape) {
      const n = newShape.reduce((a, b) => a * b, 1);
      if (n !== this.numel) throw new Error(`reshape: ${this.shape} -> ${newShape} size mismatch`);
      const t = new Tensor(this.data, newShape, this.requiresGrad, [this], (gout, goutBuf) => {
        if (!this.requiresGrad) return;
        if (goutBuf) {
          this._pendingGradBuf = goutBuf;
        } else {
          if (!this.grad) this.grad = new Float32Array(this.data.length);
          for (let i = 0; i < n; i++) this.grad[i] += gout[i];
        }
      });
      // Propagate GPU buffer as a view (shared memory, different logical shape)
      if (this.gpuBuffer) {
        t.gpuBuffer = this.gpuBuffer;
        t._dirty = this._dirty;
        t._isView = true;
      }
      return t;
    }

    /** Softmax along last axis. Input shape [..., C]. Auto-dispatches to GPU. */
    softmax() {
      const shape = this.shape;
      const C = shape[shape.length - 1];
      const rows = this.numel / C;

      const backend = gpu();
      if (backend && this.onGPU) {
        const outBuf = backend.softmax(this.gpuBuffer, rows, C);
        const t = new Tensor(new Float32Array(this.numel), [...shape], this.requiresGrad, [this], (gout, goutBuf) => {
          if (!this.requiresGrad) return;
          const be = gpu();
          if (goutBuf && be && t.gpuBuffer) {
            // softmaxBackward: d = attn * (dout - dot(dout, attn)), scale=1.0
            this._pendingGradBuf = be.softmaxBackward(t.gpuBuffer, goutBuf, rows, C, 1.0);
          } else {
            if (!this.grad) this.grad = new Float32Array(this.data.length);
            const outData = t.data;
            for (let r = 0; r < rows; r++) {
              const off = r * C;
              let dot = 0;
              for (let j = 0; j < C; j++) dot += gout[off + j] * outData[off + j];
              for (let j = 0; j < C; j++) this.grad[off + j] += outData[off + j] * (gout[off + j] - dot);
            }
          }
        });
        t.gpuBuffer = outBuf;
        t._dirty = "gpu";
        return t;
      }

      const out = new Float32Array(this.numel);
      for (let r = 0; r < rows; r++) {
        const off = r * C;
        let mx = -Infinity;
        for (let j = 0; j < C; j++) { const v = this.data[off + j]; if (v > mx) mx = v; }
        let s = 0;
        for (let j = 0; j < C; j++) { const e = Math.exp(this.data[off + j] - mx); out[off + j] = e; s += e; }
        const inv = 1.0 / Math.max(s, 1e-12);
        for (let j = 0; j < C; j++) out[off + j] *= inv;
      }
      const outData = out;
      return new Tensor(out, [...shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(this.data.length);
        for (let r = 0; r < rows; r++) {
          const off = r * C;
          let dot = 0;
          for (let j = 0; j < C; j++) dot += gout[off + j] * outData[off + j];
          for (let j = 0; j < C; j++) this.grad[off + j] += outData[off + j] * (gout[off + j] - dot);
        }
      });
    }

    /** SiLU activation: x * sigmoid(x). */
    silu() {
      const n = this.numel;
      const out = new Float32Array(n);
      const sig = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const s = 1.0 / (1.0 + Math.exp(-this.data[i]));
        sig[i] = s;
        out[i] = this.data[i] * s;
      }
      const xData = this.data;
      // Capture GPU buffer ref for backward (if x is on GPU, we can do backward on GPU)
      const xBuf = this.gpuBuffer;
      const xOnGPU = this.onGPU;
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        const backend = gpu();
        if (xOnGPU && backend && xBuf) {
          // GPU backward: silu_backward op=6, A=x, B=gout
          const goutBuf = backend.createBufferFromData(gout);
          this._pendingGradBuf = backend.elementwise(xBuf, goutBuf, n, 6);
          backend.releaseBuffer(goutBuf);
        } else {
          if (!this.grad) this.grad = new Float32Array(n);
          for (let i = 0; i < n; i++) {
            // d(silu)/dx = sig(x) * (1 + x*(1 - sig(x)))
            this.grad[i] += gout[i] * sig[i] * (1 + xData[i] * (1 - sig[i]));
          }
        }
      });
    }

    /**
     * Split along last axis into chunks of given sizes.
     * sizes: array of ints summing to shape[-1].
     * Returns array of Tensors.
     */
    splitLast(sizes) {
      const shape = this.shape;
      const C = shape[shape.length - 1];
      const totalSize = sizes.reduce((a, b) => a + b, 0);
      if (totalSize !== C) throw new Error(`splitLast: sizes ${sizes} don't sum to ${C}`);

      const rows = this.numel / C;
      const outerShape = shape.slice(0, -1);
      const results = [];
      let colOff = 0;

      for (const sz of sizes) {
        const outData = new Float32Array(rows * sz);
        for (let r = 0; r < rows; r++) {
          for (let j = 0; j < sz; j++) {
            outData[r * sz + j] = this.data[r * C + colOff + j];
          }
        }
        const myColOff = colOff;
        const mySz = sz;
        results.push(new Tensor(outData, [...outerShape, sz], this.requiresGrad, [this], (gout) => {
          if (!this.requiresGrad) return;
          if (!this.grad) this.grad = new Float32Array(this.data.length);
          for (let r = 0; r < rows; r++) {
            for (let j = 0; j < mySz; j++) {
              this.grad[r * C + myColOff + j] += gout[r * mySz + j];
            }
          }
        }));
        colOff += sz;
      }
      return results;
    }

    /**
     * Batched matmul for 3D tensors: [B, M, K] @ [B, K, N] -> [B, M, N].
     * Loops over batch dim, delegates to 2D matmul per slice.
     */
    batchedMatmul(other) {
      if (this.shape.length !== 3 || other.shape.length !== 3)
        throw new Error(`batchedMatmul requires 3D tensors, got ${this.shape} and ${other.shape}`);
      const [b1, m, k1] = this.shape;
      const [b2, k2, n] = other.shape;
      if (b1 !== b2) throw new Error(`batchedMatmul batch mismatch: ${b1} vs ${b2}`);
      if (k1 !== k2) throw new Error(`batchedMatmul inner dim mismatch: ${k1} vs ${k2}`);
      const B = b1;
      const req = this.requiresGrad || other.requiresGrad;

      const backend = gpu();
      if (backend) {
        if (this.onGPU && !other.onGPU) other.toGPU();
        else if (!this.onGPU && other.onGPU) this.toGPU();

        if (this.onGPU && other.onGPU) {
          const outBuf = backend.batchedMatmul(this.gpuBuffer, other.gpuBuffer, B, m, k1, n);
          const aBuf = this.gpuBuffer;
          const bBuf = other.gpuBuffer;
          const t = new Tensor(new Float32Array(B * m * n), [B, m, n], req, [this, other], (gout, goutBuf) => {
            const be = gpu();
            if (goutBuf && be && aBuf && bBuf) {
              // dA[b] = gout[b] @ B[b]^T
              if (this.requiresGrad) {
                this._pendingGradBuf = be.batchedMatmul(goutBuf, bBuf, B, m, n, k1, true);
              }
              // dB[b] = A[b]^T @ gout[b]
              if (other.requiresGrad) {
                other._pendingGradBuf = be.batchedMatmul(aBuf, goutBuf, B, k1, m, n, true);
              }
            } else {
              this._batchedMatmulCPUBackward(other, gout, B, m, k1, n);
            }
          });
          t.gpuBuffer = outBuf;
          t._dirty = "gpu";
          return t;
        }
      }

      const aData = this.data;
      const bData = other.data;
      const out = new Float32Array(B * m * n);
      for (let b = 0; b < B; b++) {
        const aOff = b * m * k1, bOff = b * k2 * n, oOff = b * m * n;
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            let s = 0.0;
            for (let k = 0; k < k1; k++) s += aData[aOff + i * k1 + k] * bData[bOff + k * n + j];
            out[oOff + i * n + j] = s;
          }
        }
      }
      return new Tensor(out, [B, m, n], req, [this, other], (gout) => {
        this._batchedMatmulCPUBackward(other, gout, B, m, k1, n);
      });
    }

    _batchedMatmulCPUBackward(other, gout, B, m, k, n) {
      const aData = this.data, bData = other.data;
      if (this.requiresGrad) {
        if (!this.grad) this.grad = new Float32Array(this.data.length);
        for (let b = 0; b < B; b++) {
          const gOff = b * m * n, bOff = b * k * n, aOff = b * m * k;
          for (let i = 0; i < m; i++)
            for (let kk = 0; kk < k; kk++) {
              let s = 0.0;
              for (let j = 0; j < n; j++) s += gout[gOff + i * n + j] * bData[bOff + kk * n + j];
              this.grad[aOff + i * k + kk] += s;
            }
        }
      }
      if (other.requiresGrad) {
        if (!other.grad) other.grad = new Float32Array(other.data.length);
        for (let b = 0; b < B; b++) {
          const aOff = b * m * k, gOff = b * m * n, bOff = b * k * n;
          for (let kk = 0; kk < k; kk++)
            for (let j = 0; j < n; j++) {
              let s = 0.0;
              for (let i = 0; i < m; i++) s += aData[aOff + i * k + kk] * gout[gOff + i * n + j];
              other.grad[bOff + kk * n + j] += s;
            }
        }
      }
    }

    /**
     * Transpose last two dims of a 3D tensor: [B, M, N] -> [B, N, M].
     */
    transpose3d() {
      if (this.shape.length !== 3) throw new Error(`transpose3d requires 3D, got ${this.shape}`);
      const [B, m, n] = this.shape;

      const backend = gpu();
      if (backend && this.onGPU) {
        const outBuf = backend.batchedTranspose(this.gpuBuffer, B, m, n);
        const t = new Tensor(new Float32Array(B * m * n), [B, n, m], this.requiresGrad, [this], (gout, goutBuf) => {
          if (!this.requiresGrad) return;
          const be = gpu();
          if (goutBuf && be) {
            this._pendingGradBuf = be.batchedTranspose(goutBuf, B, n, m);
          } else {
            if (!this.grad) this.grad = new Float32Array(this.data.length);
            for (let b = 0; b < B; b++) {
              const off = b * n * m;
              for (let i = 0; i < m; i++)
                for (let j = 0; j < n; j++)
                  this.grad[b * m * n + i * n + j] += gout[off + j * m + i];
            }
          }
        });
        t.gpuBuffer = outBuf;
        t._dirty = "gpu";
        return t;
      }

      const out = new Float32Array(B * m * n);
      for (let b = 0; b < B; b++) {
        const off = b * m * n;
        for (let i = 0; i < m; i++)
          for (let j = 0; j < n; j++)
            out[b * n * m + j * m + i] = this.data[off + i * n + j];
      }
      return new Tensor(out, [B, n, m], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(this.data.length);
        for (let b = 0; b < B; b++) {
          const off = b * n * m;
          for (let i = 0; i < m; i++)
            for (let j = 0; j < n; j++)
              this.grad[b * m * n + i * n + j] += gout[off + j * m + i];
        }
      });
    }

    /**
     * Apply a causal mask: set upper-triangular entries to -Infinity.
     * Input: [B, T, T] (attention scores).
     */
    causalMask() {
      const shape = this.shape;
      const T = shape[shape.length - 1];
      const rows = this.numel / T;
      const out = new Float32Array(this.numel);
      out.set(this.data);
      // For each [..., i, j] where j > i, set to -Infinity
      for (let r = 0; r < rows; r++) {
        const rowInT = r % T;  // which row within the T×T block
        const off = r * T;
        for (let j = rowInT + 1; j < T; j++) {
          out[off + j] = -Infinity;
        }
      }
      return new Tensor(out, [...shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(this.data.length);
        // Gradient is zero where we masked, pass through elsewhere
        for (let r = 0; r < rows; r++) {
          const rowInT = r % T;
          const off = r * T;
          for (let j = 0; j <= rowInT; j++) {
            this.grad[off + j] += gout[off + j];
          }
          // j > rowInT: masked, gradient = 0
        }
      });
    }

    /**
     * Embedding lookup for 2D index tensor: emb[V,D], idx[B,T] -> [B,T,D]
     * Static method for multi-token sequences.
     */
    static embeddingLookup2D(emb, idxFlat, B, T) {
      const [V, D] = emb.shape;
      const out = new Float32Array(B * T * D);
      for (let i = 0; i < B * T; i++) {
        const id = Math.max(0, Math.min(V - 1, idxFlat[i] | 0));
        const src = id * D;
        const dst = i * D;
        for (let j = 0; j < D; j++) out[dst + j] = emb.data[src + j];
      }
      return new Tensor(out, [B, T, D], emb.requiresGrad, [emb], (gout) => {
        if (!emb.requiresGrad) return;
        if (!emb.grad) emb.grad = new Float32Array(emb.data.length);
        for (let i = 0; i < B * T; i++) {
          const id = Math.max(0, Math.min(V - 1, idxFlat[i] | 0));
          const dst = id * D;
          const src = i * D;
          for (let j = 0; j < D; j++) emb.grad[dst + j] += gout[src + j];
        }
      });
    }

    static embeddingLookup2DGPU(emb, idxFlat, B, T) {
      const backend = gpu();
      const [V, D] = emb.shape;
      const BT = B * T;

      // Upload indices as u32
      const idxU32 = new Uint32Array(BT);
      for (let i = 0; i < BT; i++) idxU32[i] = Math.max(0, Math.min(V - 1, idxFlat[i] | 0));
      const idxBuf = backend.createBufferFromData(idxU32);

      // Ensure emb is on GPU
      if (!emb.gpuBuffer || emb._dirty === "cpu") emb.toGPU();

      const outBuf = backend.embeddingForward(emb.gpuBuffer, idxBuf, BT, D, V);
      idxBuf.destroy();

      // Create output tensor with GPU buffer, backward stays CPU (scatter-add is small)
      const outData = new Float32Array(BT * D); // placeholder — will be overwritten by toCPU if needed
      const t = new Tensor(outData, [B, T, D], emb.requiresGrad, [emb], (gout) => {
        if (!emb.requiresGrad) return;
        if (!emb.grad) emb.grad = new Float32Array(emb.data.length);
        for (let i = 0; i < BT; i++) {
          const id = Math.max(0, Math.min(V - 1, idxFlat[i] | 0));
          const dst = id * D;
          const src = i * D;
          for (let j = 0; j < D; j++) emb.grad[dst + j] += gout[src + j];
        }
      });
      t.gpuBuffer = outBuf;
      t._dirty = "gpu";
      return t;
    }

    /** Unified embedding lookup — auto-dispatches to GPU or CPU. */
    static embedding(emb, idxFlat, B, T) {
      const backend = gpu();
      if (backend && emb.onGPU) {
        return Tensor.embeddingLookup2DGPU(emb, idxFlat, B, T);
      }
      return Tensor.embeddingLookup2D(emb, idxFlat, B, T);
    }

    transpose2d() {
      const [m, n] = this.shape;
      const out = new Float32Array(m * n);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) out[j * m + i] = this.data[i * n + j];
      }
      const req = this.requiresGrad;

      // If on GPU, also create transposed GPU buffer
      let outBuf = null;
      let dirty = "cpu";
      if (this.onGPU && gpu()) {
        outBuf = gpu().createBufferFromData(out);
        dirty = "gpu";
      }

      return new Tensor(
        out,
        [n, m],
        req,
        [this],
        (gout) => {
          if (!this.requiresGrad) return;
          if (!this.grad) this.grad = new Float32Array(this.data.length);
          for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) this.grad[i * n + j] += gout[j * m + i];
          }
        },
        outBuf,
        dirty
      );
    }

    /** Backward pass. For GPU tensors, reads back loss scalar first. */
    backward() {
      if (this.shape.length !== 0 && !(this.shape.length === 1 && this.shape[0] === 1)) {
        throw new Error("backward() expects scalar loss tensor");
      }
      const topo = [];
      const seen = new Set();
      const build = (t) => {
        if (seen.has(t)) return;
        seen.add(t);
        for (const p of t._parents) build(p);
        topo.push(t);
      };
      build(this);

      this.grad = new Float32Array([1.0]);
      for (let i = topo.length - 1; i >= 0; i--) {
        const t = topo[i];
        if (!t.grad) continue;
        // Pass both CPU grad array and GPU grad buffer (if available)
        t._backward(t.grad, t._gradGPUBuf || null);
      }
    }

    /** Async backward: resolves any pending GPU grad buffers to CPU. */
    async resolveGrads() {
      const topo = [];
      const seen = new Set();
      const build = (t) => {
        if (seen.has(t)) return;
        seen.add(t);
        for (const p of t._parents) build(p);
        topo.push(t);
      };
      build(this);

      const backend = gpu();
      for (const t of topo) {
        if (t._pendingGradBuf && backend) {
          const gradData = await backend.readBuffer(t._pendingGradBuf, t.numel);
          if (!t.grad) t.grad = new Float32Array(t.data.length);
          for (let i = 0; i < gradData.length; i++) t.grad[i] += gradData[i];
          backend.releaseBuffer(t._pendingGradBuf);
          t._pendingGradBuf = null;
        }
        // Handle RMSNorm weight grad: [rows*d] per-row contributions → sum to [d]
        if (t._pendingDWBuf && backend) {
          const rows = t._pendingDWRows;
          const d = t._pendingDWD;
          const dwData = await backend.readBuffer(t._pendingDWBuf, rows * d);
          if (!t.grad) t.grad = new Float32Array(d);
          for (let r = 0; r < rows; r++) {
            for (let j = 0; j < d; j++) {
              t.grad[j] += dwData[r * d + j];
            }
          }
          backend.releaseBuffer(t._pendingDWBuf);
          t._pendingDWBuf = null;
          t._pendingDWRows = 0;
          t._pendingDWD = 0;
        }
      }
    }

    /** Like resolveGrads() but keeps gradients on GPU as _gradGPUBuf. */
    async resolveGradsGPU() {
      const topo = [];
      const seen = new Set();
      const build = (t) => {
        if (seen.has(t)) return;
        seen.add(t);
        for (const p of t._parents) build(p);
        topo.push(t);
      };
      build(this);

      const backend = gpu();
      if (!backend) { await this.resolveGrads(); return; }

      for (const t of topo) {
        if (t._pendingGradBuf) {
          if (!t._gradGPUBuf) {
            t._gradGPUBuf = t._pendingGradBuf;
          } else {
            // Accumulate via elementwise add (op=0)
            const acc = backend.elementwise(t._gradGPUBuf, t._pendingGradBuf, t.numel, 0);
            backend.releaseBuffer(t._gradGPUBuf);
            backend.releaseBuffer(t._pendingGradBuf);
            t._gradGPUBuf = acc;
          }
          t._pendingGradBuf = null;
        }
        // RMSNorm weight grad: [rows*d] → reduce rows → [d] (small, keep on CPU then re-upload)
        if (t._pendingDWBuf) {
          const rows = t._pendingDWRows;
          const d = t._pendingDWD;
          const dwData = await backend.readBuffer(t._pendingDWBuf, rows * d);
          const acc = new Float32Array(d);
          for (let r = 0; r < rows; r++) {
            for (let j = 0; j < d; j++) acc[j] += dwData[r * d + j];
          }
          backend.releaseBuffer(t._pendingDWBuf);
          t._pendingDWBuf = null;
          t._pendingDWRows = 0;
          t._pendingDWD = 0;
          const dwGPUBuf = backend.createBufferFromData(acc);
          if (!t._gradGPUBuf) {
            t._gradGPUBuf = dwGPUBuf;
          } else {
            const merged = backend.elementwise(t._gradGPUBuf, dwGPUBuf, d, 0);
            backend.releaseBuffer(t._gradGPUBuf);
            dwGPUBuf.destroy();
            t._gradGPUBuf = merged;
          }
        }
      }
    }

    /**
     * Unified cross-entropy loss — auto-dispatches to GPU or CPU.
     * logits: Tensor [N, V], targets: Int32Array [N]. Returns scalar loss Tensor.
     */
    static async crossEntropy(logits, targets) {
      const backend = gpu();
      if (backend && logits.onGPU) {
        return crossEntropyGPU(logits, targets);
      }
      return crossEntropy(logits, targets);
    }

    /**
     * Unified RMS normalization — auto-dispatches (backward uses GPU when available).
     * x: Tensor, w: Tensor or number (eps), eps: number.
     */
    static rmsNorm(x, epsOrW = 1e-6, eps = 1e-6) {
      return rmsNorm(x, epsOrW, eps);
    }
  }

  function crossEntropy(logits, targets) {
    // logits: [N, V], targets: Int32Array length N
    const [n, v] = logits.shape;
    if (targets.length !== n) throw new Error("targets length mismatch");
    const x = logits.data;
    let loss = 0.0;
    const probs = new Float32Array(n * v);
    for (let i = 0; i < n; i++) {
      let maxv = -Infinity;
      const row = i * v;
      for (let j = 0; j < v; j++) {
        const val = x[row + j];
        if (val > maxv) maxv = val;
      }
      let s = 0.0;
      for (let j = 0; j < v; j++) {
        const e = Math.exp(x[row + j] - maxv);
        probs[row + j] = e;
        s += e;
      }
      const inv = 1.0 / Math.max(s, 1e-12);
      for (let j = 0; j < v; j++) probs[row + j] *= inv;
      const p = Math.max(probs[row + targets[i]], 1e-12);
      loss += -Math.log(p);
    }
    loss /= n;

    return new Tensor(
      new Float32Array([loss]),
      [],
      logits.requiresGrad,
      [logits],
      (gout) => {
        if (!logits.requiresGrad) return;
        if (!logits.grad) logits.grad = new Float32Array(logits.data.length);
        const scale = (gout[0] || 1.0) / n;
        for (let i = 0; i < n; i++) {
          const row = i * v;
          for (let j = 0; j < v; j++) logits.grad[row + j] += probs[row + j] * scale;
          logits.grad[row + targets[i]] -= scale;
        }
      }
    );
  }

  /**
   * GPU cross-entropy: runs softmax + NLL on GPU, backward produces GPU grad buffer.
   * logits can be on GPU (gpuBuffer) or CPU (data). targets is Int32Array.
   */
  async function crossEntropyGPU(logits, targets) {
    const backend = gpu();
    const [n, v] = logits.shape;
    if (targets.length !== n) throw new Error("targets length mismatch");

    // Ensure logits are on GPU
    let logitsBuf = logits.gpuBuffer && logits._dirty !== "cpu" ? logits.gpuBuffer : null;
    let ownLogitsBuf = false;
    if (!logitsBuf) {
      logitsBuf = backend.createBufferFromData(logits.data);
      ownLogitsBuf = true;
    }

    // Upload targets as u32 buffer
    const targetsU32 = new Uint32Array(targets.length);
    for (let i = 0; i < targets.length; i++) targetsU32[i] = targets[i] >>> 0;
    const targetsBuf = backend.createBufferFromData(targetsU32);

    // Forward: softmax + NLL on GPU
    const { probsBuf, lossesBuf } = backend.crossEntropyForward(logitsBuf, targetsBuf, n, v);

    // Read back per-sample losses (N floats — small) and compute mean
    const lossData = await backend.readBuffer(lossesBuf, n);
    backend.releaseBuffer(lossesBuf);
    let lossSum = 0;
    for (let i = 0; i < n; i++) lossSum += lossData[i];
    const meanLoss = lossSum / n;

    if (ownLogitsBuf) logitsBuf.destroy();

    // Return scalar loss tensor with GPU backward
    return new Tensor(
      new Float32Array([meanLoss]),
      [],
      logits.requiresGrad,
      [logits],
      (_gout) => {
        if (!logits.requiresGrad) return;
        const scale = 1.0 / n;
        const dLogitsBuf = backend.crossEntropyBackward(probsBuf, targetsBuf, n, v, scale);
        logits._pendingGradBuf = dLogitsBuf;
        // probsBuf/targetsBuf freed when resolveGrads/resolveGradsGPU runs
      }
    );
  }

  function embeddingLookup(emb, idx) {
    // emb: [V, D], idx: Int32Array [B] => out: [B, D]
    const [v, d] = emb.shape;
    const b = idx.length;
    const out = new Float32Array(b * d);
    for (let i = 0; i < b; i++) {
      const id = Math.max(0, Math.min(v - 1, idx[i] | 0));
      const src = id * d;
      const dst = i * d;
      for (let j = 0; j < d; j++) out[dst + j] = emb.data[src + j];
    }
    const req = emb.requiresGrad;

    // If emb is on GPU, put output on GPU too
    let outBuf = null;
    let dirty = "cpu";
    if (emb.onGPU && gpu()) {
      outBuf = gpu().createBufferFromData(out);
      dirty = "gpu";
    }

    return new Tensor(
      out,
      [b, d],
      req,
      [emb],
      (gout) => {
        if (!emb.requiresGrad) return;
        if (!emb.grad) emb.grad = new Float32Array(emb.data.length);
        for (let i = 0; i < b; i++) {
          const id = Math.max(0, Math.min(v - 1, idx[i] | 0));
          const dst = id * d;
          const src = i * d;
          for (let j = 0; j < d; j++) emb.grad[dst + j] += gout[src + j];
        }
      },
      outBuf,
      dirty
    );
  }

  /**
   * RMSNorm: works on last dim. Supports [B, D] or [B, T, D].
   * Optional weight w: [D] applied as elementwise scale after normalization.
   */
  function rmsNorm(x, epsOrW = 1e-6, eps = 1e-6) {
    let w = null;
    if (typeof epsOrW === "object" && epsOrW instanceof Tensor) {
      w = epsOrW;  // rmsNorm(x, wTensor, eps)
    } else {
      eps = epsOrW; // rmsNorm(x, eps)
    }

    const shape = x.shape;
    const d = shape[shape.length - 1];
    const rows = x.numel / d;
    const out = new Float32Array(x.numel);
    const inv = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
      const row = i * d;
      let s2 = 0.0;
      for (let j = 0; j < d; j++) {
        const v = x.data[row + j];
        s2 += v * v;
      }
      const r = 1.0 / Math.sqrt(s2 / d + eps);
      inv[i] = r;
      for (let j = 0; j < d; j++) {
        let val = x.data[row + j] * r;
        if (w) val *= w.data[j];
        out[row + j] = val;
      }
    }
    const req = x.requiresGrad || (w && w.requiresGrad);
    const parents = w ? [x, w] : [x];
    // Capture GPU state for backward
    const xBuf = x.gpuBuffer;
    const wBuf = w ? w.gpuBuffer : null;
    const xOnGPU = x.onGPU;

    return new Tensor(
      out,
      [...shape],
      req,
      parents,
      (gout) => {
        const backend = gpu();
        if (xOnGPU && backend && xBuf) {
          // GPU backward path
          const goutBuf = backend.createBufferFromData(gout);
          const result = backend.rmsNormBackward(xBuf, goutBuf, wBuf, rows, d, eps);
          if (x.requiresGrad) {
            x._pendingGradBuf = result.dXBuf;
          } else {
            backend.releaseBuffer(result.dXBuf);
          }
          if (w && w.requiresGrad) {
            // dWBuf is [rows*d] — per-row contributions, need to sum across rows on CPU
            // We'll store it as pending and reduce in resolveGrads
            w._pendingDWBuf = result.dWBuf;
            w._pendingDWRows = rows;
            w._pendingDWD = d;
          } else {
            backend.releaseBuffer(result.dWBuf);
          }
          backend.releaseBuffer(goutBuf);
        } else {
          // CPU backward path
          if (x.requiresGrad) {
            if (!x.grad) x.grad = new Float32Array(x.data.length);
            for (let i = 0; i < rows; i++) {
              const row = i * d;
              const r = inv[i];
              let dot = 0.0;
              for (let j = 0; j < d; j++) {
                const gj = w ? gout[row + j] * w.data[j] : gout[row + j];
                dot += gj * x.data[row + j];
              }
              const coeff = (r * r * r * dot) / d;
              for (let j = 0; j < d; j++) {
                const xi = x.data[row + j];
                const gj = w ? gout[row + j] * w.data[j] : gout[row + j];
                x.grad[row + j] += gj * r - xi * coeff;
              }
            }
          }
          if (w && w.requiresGrad) {
            if (!w.grad) w.grad = new Float32Array(w.data.length);
            for (let i = 0; i < rows; i++) {
              const row = i * d;
              const r = inv[i];
              for (let j = 0; j < d; j++) {
                w.grad[j] += gout[row + j] * x.data[row + j] * r;
              }
            }
          }
        }
      }
    );
  }

  // --- f16 ↔ f32 conversion utilities ---

  /**
   * Convert Float32Array to Uint16Array of IEEE 754 half-precision floats.
   * Handles Inf, NaN, denorms, and rounding.
   */
  function f32ToF16(f32arr) {
    const out = new Uint16Array(f32arr.length);
    const view = new DataView(new ArrayBuffer(4));
    for (let i = 0; i < f32arr.length; i++) {
      view.setFloat32(0, f32arr[i], true);
      const bits = view.getUint32(0, true);
      const sign = (bits >>> 31) & 1;
      const exp = (bits >>> 23) & 0xFF;
      const frac = bits & 0x7FFFFF;

      let h;
      if (exp === 0xFF) {
        // Inf / NaN
        h = (sign << 15) | 0x7C00 | (frac ? 0x0200 : 0);
      } else if (exp > 142) {
        // Overflow → Inf
        h = (sign << 15) | 0x7C00;
      } else if (exp < 103) {
        // Underflow → zero
        h = sign << 15;
      } else if (exp < 113) {
        // Denormalized f16
        const m = (0x800000 | frac) >>> (126 - exp);
        h = (sign << 15) | (m >>> 13);
      } else {
        // Normalized
        h = (sign << 15) | ((exp - 112) << 10) | (frac >>> 13);
      }
      out[i] = h;
    }
    return out;
  }

  /**
   * Convert Uint16Array of IEEE 754 half-precision floats to Float32Array.
   */
  function f16ToF32(u16arr) {
    const out = new Float32Array(u16arr.length);
    const view = new DataView(new ArrayBuffer(4));
    for (let i = 0; i < u16arr.length; i++) {
      const h = u16arr[i];
      const sign = (h >>> 15) & 1;
      const exp = (h >>> 10) & 0x1F;
      const frac = h & 0x3FF;

      let bits;
      if (exp === 0x1F) {
        // Inf / NaN
        bits = (sign << 31) | 0x7F800000 | (frac << 13);
      } else if (exp === 0) {
        if (frac === 0) {
          bits = sign << 31;
        } else {
          // Denorm → normalize
          let e = -1;
          let f = frac;
          do { e++; f <<= 1; } while ((f & 0x400) === 0);
          bits = (sign << 31) | ((112 - e) << 23) | ((f & 0x3FF) << 13);
        }
      } else {
        bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
      }
      view.setUint32(0, bits, true);
      out[i] = view.getFloat32(0, true);
    }
    return out;
  }

  /**
   * Encode Float32Array as base64-encoded f16 binary string.
   * Result is ~1/6 the size of JSON number arrays.
   */
  function weightsToF16Base64(f32arr) {
    const f16 = f32ToF16(f32arr);
    const bytes = new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  /**
   * Decode base64-encoded f16 binary string back to Float32Array.
   */
  function f16Base64ToWeights(b64str) {
    const binary = atob(b64str);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
    return f16ToF32(u16);
  }

  /**
   * Encode weight delta as int8 base64.
   * Returns { delta_b64, delta_scale, count } or null if no change.
   */
  function encodeInt8Delta(updated, original) {
    const n = updated.length;
    let maxAbs = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(updated[i] - original[i]);
      if (d > maxAbs) maxAbs = d;
    }
    if (maxAbs === 0) return null;
    const scale = maxAbs / 127;
    const invScale = 127 / maxAbs;
    const int8 = new Int8Array(n);
    for (let i = 0; i < n; i++) {
      int8[i] = Math.max(-127, Math.min(127, Math.round((updated[i] - original[i]) * invScale)));
    }
    const bytes = new Uint8Array(int8.buffer, int8.byteOffset, int8.byteLength);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return { delta_b64: btoa(binary), delta_scale: scale, count: n };
  }

  /**
   * Decode int8 delta base64 and apply to original weights.
   */
  function decodeInt8Delta(b64, scale, original) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const int8 = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Float32Array(int8.length);
    for (let i = 0; i < int8.length; i++) out[i] = original[i] + int8[i] * scale;
    return out;
  }

  /**
   * Optimizer — encapsulates Adam/SGD with GPU/CPU auto-dispatch.
   * Handles gradient resolution, clipping, and parameter updates.
   */
  class Optimizer {
    constructor(params, opts = {}) {
      this.params = params;
      this.lr = opts.lr || 5e-4;
      this.type = (opts.optimizer || "adam").toLowerCase();
      this.beta1 = opts.beta1 || 0.9;
      this.beta2 = opts.beta2 || 0.999;
      this.eps = opts.eps || 1e-8;
      this.weightDecay = opts.weightDecay || 0.0;
      this.momentum = opts.momentum || 0.0;
      this.nesterov = !!opts.nesterov;
      this.tcoef = opts.tcoef != null ? opts.tcoef : (opts.optimizer === "lars" ? 0.001 : 0.0);
      this.nsSteps = opts.nsSteps || 0;
      this.nsCoeffs = opts.nsCoeffs || null;
      this.preWd = opts.preWd !== undefined ? !!opts.preWd : true;
      this.classic = opts.classic !== undefined ? !!opts.classic : true;
      this.maxNorm = opts.gradClipNorm != null ? opts.gradClipNorm : 1.0;
      this.warmupSteps = Math.max(1, opts.warmupSteps || 10);
      this._step = window._adamStep || 0;
      this._backend = gpu();
      this._useGPU = !!this._backend;
      this._initState();
    }

    _initState() {
      const gpuAdam = this._useGPU && (this.type === "adam" || this.type === "adamw");
      if (gpuAdam) {
        const be = this._backend;
        for (const p of this.params) {
          const bytes = p.numel * 4;
          p._adamMBuf = be.createEmptyBuffer(bytes);
          p._adamVBuf = be.createEmptyBuffer(bytes);
          be.writeBuffer(p._adamMBuf, new Float32Array(p.numel));
          be.writeBuffer(p._adamVBuf, new Float32Array(p.numel));
        }
      }
    }

    /** Resolve gradients (GPU or CPU) and apply gradient clipping. */
    async resolveAndClip(loss) {
      const be = this._backend;
      if (this._useGPU && be) {
        await loss.resolveGradsGPU();
        if (this.maxNorm > 0) {
          be.beginBatch();
          const normBufs = this.params.map(p =>
            p._gradGPUBuf ? be.reduce(p._gradGPUBuf, 1, p.numel, 2) : null
          );
          be.endBatch();
          const normSqs = await Promise.all(
            normBufs.map(buf => buf
              ? be.readBuffer(buf, 1).then(d => { be.releaseBuffer(buf); return d[0]; })
              : Promise.resolve(0)
            )
          );
          const totalNorm = Math.sqrt(normSqs.reduce((a, b) => a + b, 0));
          if (totalNorm > this.maxNorm) {
            const clipScale = this.maxNorm / totalNorm;
            be.beginBatch();
            const scaledBufs = this.params.map(p => {
              if (!p._gradGPUBuf) return null;
              return be.elementwise(p._gradGPUBuf, null, p.numel, 3, clipScale);
            });
            be.endBatch();
            this.params.forEach((p, i) => {
              if (scaledBufs[i]) {
                be.releaseBuffer(p._gradGPUBuf);
                p._gradGPUBuf = scaledBufs[i];
              }
            });
          }
        }
      } else {
        await loss.resolveGrads();
        if (this.maxNorm > 0) {
          let totalNormSq = 0;
          for (const p of this.params) {
            if (p.grad) for (let i = 0; i < p.grad.length; i++) totalNormSq += p.grad[i] * p.grad[i];
          }
          const totalNorm = Math.sqrt(totalNormSq);
          if (totalNorm > this.maxNorm) {
            const clipScale = this.maxNorm / totalNorm;
            for (const p of this.params) {
              if (p.grad) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= clipScale;
            }
          }
        }
      }
    }

    /** Perform one optimizer step (Adam or SGD, GPU or CPU). */
    async step() {
      this._step++;
      window._adamStep = this._step;
      const s = this._step;
      const effLr = s <= this.warmupSteps ? this.lr * (s / this.warmupSteps) : this.lr;
      const bc1 = 1 - Math.pow(this.beta1, s);
      const bc2 = 1 - Math.pow(this.beta2, s);
      const be = this._backend;

      if (this._useGPU && be && (this.type === "adam" || this.type === "adamw") && this.momentum === 0) {
        be.beginBatch();
        for (const p of this.params) {
          if (!p._gradGPUBuf) continue;
          if (this.type === "adamw") {
            be.adamwUpdate(p._gradGPUBuf, p._adamMBuf, p._adamVBuf, p.gpuBuffer,
              p.numel, this.beta1, this.beta2, effLr, bc1, bc2, this.eps, this.weightDecay);
          } else {
            be.adamUpdate(p._gradGPUBuf, p._adamMBuf, p._adamVBuf, p.gpuBuffer,
              p.numel, this.beta1, this.beta2, effLr, bc1, bc2, this.eps);
          }
        }
        be.endBatch();
        for (const p of this.params) {
          if (p._gradGPUBuf) { be.releaseBuffer(p._gradGPUBuf); p._gradGPUBuf = null; }
        }
      } else if (this._useGPU && be && this.type === "sgd" && this.momentum === 0 && this.weightDecay === 0) {
        be.beginBatch();
        for (const p of this.params) {
          if (!p._gradGPUBuf) continue;
          be.sgdUpdate(p._gradGPUBuf, p.gpuBuffer, p.numel, this.lr);
        }
        be.endBatch();
        for (const p of this.params) {
          if (p._gradGPUBuf) { be.releaseBuffer(p._gradGPUBuf); p._gradGPUBuf = null; }
        }
      } else {
        // CPU path (also used for momentum / weight-decay SGD)
        if (this.type === "adam" || this.type === "adamw" || this.type === "lamb") {
          for (const p of this.params) {
            if (!p.grad) continue;
            if (!p._adam_m) p._adam_m = new Float32Array(p.data.length);
            if (!p._adam_v) p._adam_v = new Float32Array(p.data.length);
            const m = p._adam_m, v = p._adam_v, g = p.grad;
            const wd = this.type === "adam" ? 0 : this.weightDecay;
            let r1 = 0, r2 = 0;
            const ups = this.type === "lamb" ? new Float32Array(p.data.length) : null;
            for (let i = 0; i < p.data.length; i++) {
              m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i];
              v[i] = this.beta2 * v[i] + (1 - this.beta2) * g[i] * g[i];
              const adamDir = (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + this.eps);
              const up = adamDir + wd * p.data[i];
              if (ups) {
                ups[i] = up;
                r1 += p.data[i] * p.data[i];
                r2 += up * up;
              } else {
                p.data[i] -= effLr * up;
              }
            }
            if (ups) {
              r1 = Math.sqrt(r1); r2 = Math.sqrt(r2);
              const r = (r1 > 0 && r2 > 0) ? r1 / r2 : 1;
              for (let i = 0; i < p.data.length; i++) p.data[i] -= effLr * r * ups[i];
            }
            if (typeof p.markCPUDirty === "function") p.markCPUDirty();
          }
        } else if (this.type === "lars" || this.type === "muon") {
          for (const p of this.params) {
            if (!p.grad) continue;
            const g = Float32Array.from(p.grad);
            if (this.preWd && this.weightDecay > 0) {
              for (let i = 0; i < p.data.length; i++) g[i] += this.weightDecay * p.data[i];
            }
            let r = 1;
            if (this.tcoef !== 0) {
              let r1 = 0, r2 = 0;
              for (let i = 0; i < p.data.length; i++) {
                r1 += p.data[i] * p.data[i];
                r2 += g[i] * g[i];
              }
              r1 = Math.sqrt(r1); r2 = Math.sqrt(r2);
              if (r1 > 0 && r2 > 0) r = this.tcoef * r1 / (r2 + this.weightDecay * r1);
            }
            if (this.classic) {
              for (let i = 0; i < g.length; i++) g[i] *= r * this.lr;
            }
            if (this.momentum > 0) {
              if (!p._sgd_b) p._sgd_b = new Float32Array(p.data.length);
              const b = p._sgd_b;
              for (let i = 0; i < g.length; i++) {
                b[i] = this.momentum * b[i] + g[i];
                g[i] = this.nesterov ? (g[i] + this.momentum * b[i]) : b[i];
              }
            }
            if (this.nsCoeffs && this.nsSteps > 0 && p.shape.length >= 1) {
              const rows = p.shape[0];
              const cols = p.numel / rows;
              const ns = newtonSchulz(g, rows, cols, this.nsSteps, this.nsCoeffs);
              g.set(ns);
            }
            if (!this.classic) {
              for (let i = 0; i < g.length; i++) g[i] *= r * this.lr;
            }
            if (!this.preWd && this.weightDecay > 0) {
              for (let i = 0; i < g.length; i++) g[i] += this.weightDecay * this.lr * p.data[i];
            }
            for (let i = 0; i < p.data.length; i++) p.data[i] -= g[i];
            if (typeof p.markCPUDirty === "function") p.markCPUDirty();
          }
        } else {
          for (const p of this.params) {
            if (!p.grad) continue;
            const g = p.grad;
            if (this.weightDecay > 0) {
              for (let i = 0; i < p.data.length; i++) g[i] += this.weightDecay * p.data[i];
            }
            if (this.momentum > 0) {
              if (!p._sgd_b) p._sgd_b = new Float32Array(p.data.length);
              const b = p._sgd_b;
              for (let i = 0; i < p.data.length; i++) {
                b[i] = this.momentum * b[i] + g[i];
                const step = this.nesterov ? (g[i] + this.momentum * b[i]) : b[i];
                p.data[i] -= this.lr * step;
              }
            } else {
              for (let i = 0; i < p.data.length; i++) p.data[i] -= this.lr * g[i];
            }
            if (typeof p.markCPUDirty === "function") p.markCPUDirty();
          }
        }
        for (const p of this.params) {
          if (p.gpuBuffer) p.toGPU();
        }
      }
    }

    /** Release optimizer state (GPU m/v buffers, CPU arrays). */
    release() {
      const be = this._backend;
      for (const p of this.params) {
        if (p._adamMBuf && be) { be.releaseBuffer(p._adamMBuf); p._adamMBuf = null; }
        if (p._adamVBuf && be) { be.releaseBuffer(p._adamVBuf); p._adamVBuf = null; }
        if (p._gradGPUBuf && be) { be.releaseBuffer(p._gradGPUBuf); p._gradGPUBuf = null; }
        p._adam_m = null;
        p._adam_v = null;
        p._sgd_b = null;
      }
    }
  }

  function Adam(params, opts = {}) {
    return new Optimizer(params, Object.assign({ optimizer: "adam" }, opts));
  }
  function AdamW(params, opts = {}) {
    return new Optimizer(params, Object.assign({ optimizer: "adamw", weightDecay: 0.01 }, opts));
  }
  function SGD(params, opts = {}) {
    return new Optimizer(params, Object.assign({ optimizer: "sgd" }, opts));
  }
  function LAMB(params, opts = {}) {
    return new Optimizer(params, Object.assign({ optimizer: "lamb", weightDecay: 0.01 }, opts));
  }
  function LARS(params, opts = {}) {
    return new Optimizer(params, Object.assign({ optimizer: "lars", momentum: 0.9, weightDecay: 1e-4, tcoef: 0.001 }, opts));
  }
  function Muon(params, opts = {}) {
    return new Optimizer(params, Object.assign({
      optimizer: "muon",
      lr: 0.001,
      momentum: 0.95,
      weightDecay: 0.1,
      nesterov: true,
      classic: false,
      preWd: false,
      tcoef: 0,
      nsSteps: 5,
      nsCoeffs: [3.4445, -4.775, 2.0315],
    }, opts));
  }

  class OptimizerGroup {
    constructor(...optimizers) {
      this.optimizers = optimizers;
    }
    async resolveAndClip(loss) {
      if (this.optimizers[0]) await this.optimizers[0].resolveAndClip(loss);
    }
    async step() {
      for (const o of this.optimizers) await o.step();
    }
    release() {
      for (const o of this.optimizers) o.release();
    }
  }

  // --- Module system (tinygrad-style nn) ---

  class Module {
    /** Recursively collect all Tensor parameters with requiresGrad. */
    parameters() {
      const params = [];
      const seen = new Set();
      const collect = (obj) => {
        if (seen.has(obj)) return;
        seen.add(obj);
        if (obj instanceof Tensor) {
          if (obj.requiresGrad) params.push(obj);
          return;
        }
        if (obj instanceof Module) {
          for (const key of Object.keys(obj)) {
            const val = obj[key];
            if (val instanceof Tensor && val.requiresGrad) {
              params.push(val);
            } else if (val instanceof Module) {
              collect(val);
            } else if (Array.isArray(val)) {
              for (const item of val) collect(item);
            }
          }
        }
      };
      collect(this);
      return params;
    }

    /** Return named parameter dict: { "layers.0.weight": Tensor, ... } */
    stateDict(prefix = "") {
      const dict = {};
      for (const key of Object.keys(this)) {
        const val = this[key];
        const name = prefix ? `${prefix}.${key}` : key;
        if (val instanceof Tensor && val.requiresGrad) {
          dict[name] = val;
        } else if (val instanceof Module) {
          Object.assign(dict, val.stateDict(name));
        } else if (Array.isArray(val)) {
          for (let i = 0; i < val.length; i++) {
            const item = val[i];
            if (item instanceof Tensor && item.requiresGrad) {
              dict[`${name}.${i}`] = item;
            } else if (item instanceof Module) {
              Object.assign(dict, item.stateDict(`${name}.${i}`));
            }
          }
        }
      }
      return dict;
    }

    /** Load parameters from a named dict (matching keys from stateDict). */
    loadStateDict(dict) {
      const myDict = this.stateDict();
      for (const [key, tensor] of Object.entries(myDict)) {
        if (dict[key]) {
          const src = dict[key];
          const srcData = src instanceof Tensor ? src.data : src;
          tensor.data.set(srcData);
          tensor.markCPUDirty();
        }
      }
    }

    /** Load from a flat weight array (order must match parameters()). */
    loadFlatWeights(flatWeights) {
      const params = this.parameters();
      let cursor = 0;
      for (const p of params) {
        const size = p.numel;
        p.data.set(flatWeights.subarray(cursor, cursor + size));
        p.markCPUDirty();
        cursor += size;
      }
      return cursor;
    }

    /** Serialize all parameters to a flat Float32Array. */
    toFlatWeights() {
      const params = this.parameters();
      let total = 0;
      for (const p of params) total += p.numel;
      const out = new Float32Array(total);
      let cursor = 0;
      for (const p of params) {
        out.set(p.data, cursor);
        cursor += p.numel;
      }
      return out;
    }

    /** Upload all parameters to GPU. */
    toGPU() {
      for (const p of this.parameters()) p.toGPU();
      return this;
    }

    /** Release all GPU buffers. */
    releaseGPU() {
      for (const p of this.parameters()) p.releaseGPU();
    }

    /** Read all parameters back to CPU. */
    async toCPU() {
      await Promise.all(this.parameters().map(p => p.toCPU()));
      return this;
    }

    /** Total parameter count. */
    paramCount() {
      let n = 0;
      for (const p of this.parameters()) n += p.numel;
      return n;
    }
  }

  /** Linear layer: y = x @ weight (no bias by default). */
  class Linear extends Module {
    constructor(inFeatures, outFeatures, { bias = false, requiresGrad = true } = {}) {
      super();
      this.inFeatures = inFeatures;
      this.outFeatures = outFeatures;
      this.weight = new Tensor(new Float32Array(inFeatures * outFeatures), [inFeatures, outFeatures], requiresGrad);
      if (bias) {
        this.bias = new Tensor(new Float32Array(outFeatures), [outFeatures], requiresGrad);
      }
    }

    forward(x) {
      let out = x.matmul(this.weight);
      if (this.bias) out = out.add(this.bias);
      return out;
    }
  }

  /** Embedding layer: lookup rows from weight table. */
  class Embedding extends Module {
    constructor(numEmbeddings, embeddingDim, { requiresGrad = true } = {}) {
      super();
      this.numEmbeddings = numEmbeddings;
      this.embeddingDim = embeddingDim;
      this.weight = new Tensor(new Float32Array(numEmbeddings * embeddingDim), [numEmbeddings, embeddingDim], requiresGrad);
    }

    forward(idxFlat, B, T) {
      return Tensor.embedding(this.weight, idxFlat, B, T);
    }
  }

  /** RMSNorm layer: normalize last dim, scale by learnable weight. */
  class RMSNorm extends Module {
    constructor(dim, { eps = 1e-6, requiresGrad = true } = {}) {
      super();
      this.dim = dim;
      this.eps = eps;
      this.weight = Tensor.fromArray(new Float32Array(dim).fill(1.0), [dim], requiresGrad);
    }

    forward(x) {
      return Tensor.rmsNorm(x, this.weight, this.eps);
    }
  }

  /** LayerNorm over the last dim. Matches tinygrad nn.LayerNorm for 1D normalized_shape. */
  class LayerNorm extends Module {
    constructor(normalizedShape, { eps = 1e-5, elementwiseAffine = true, requiresGrad = true } = {}) {
      super();
      this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
      this.eps = eps;
      const dim = this.normalizedShape[this.normalizedShape.length - 1];
      this.weight = elementwiseAffine ? Tensor.fromArray(new Float32Array(dim).fill(1.0), [dim], requiresGrad) : null;
      this.bias = elementwiseAffine ? Tensor.zeros([dim], requiresGrad) : null;
    }

    forward(x) {
      const y = x.layerNorm(this.eps);
      if (!this.weight) return y;
      return y.affineLast(this.weight, this.bias);
    }
  }

  /** Dropout module. Active only when tinygradV0.training is true. */
  class Dropout extends Module {
    constructor(p = 0.5) {
      super();
      this.p = p;
    }
    forward(x) {
      return x.dropout(this.p);
    }
  }

  class Conv2d extends Module {
    constructor(inChannels, outChannels, kernelSize, { stride = 1, padding = 0, dilation = 1, groups = 1, bias = true } = {}) {
      super();
      const kH = Array.isArray(kernelSize) ? kernelSize[0] : kernelSize;
      const kW = Array.isArray(kernelSize) ? kernelSize[1] || kernelSize[0] : kernelSize;
      this.stride = stride;
      this.padding = padding;
      this.dilation = dilation;
      this.groups = groups;
      this.weight = new Tensor(new Float32Array(outChannels * (inChannels / groups) * kH * kW), [outChannels, inChannels / groups, kH, kW], true);
      this.bias = bias ? Tensor.zeros([outChannels], true) : null;
    }
    forward(x) {
      return x.conv2d(this.weight, this.bias, this.groups, this.stride, this.dilation, this.padding);
    }
  }

  function Conv1d(inChannels, outChannels, kernelSize, opts = {}) {
    const conv = new Conv2d(inChannels, outChannels, [1, kernelSize], opts);
    const orig = conv.forward.bind(conv);
    conv.forward = (x) => {
      if (x.shape.length !== 3) throw new Error("Conv1d expects [N,C,W]");
      const [N, C, W] = x.shape;
      const y = orig(x.reshape([N, C, 1, W]));
      return y.reshape([y.shape[0], y.shape[1], y.shape[3]]);
    };
    return conv;
  }

  class ConvTranspose2d extends Module {
    constructor(inChannels, outChannels, kernelSize, { stride = 1, padding = 0, outputPadding = 0, dilation = 1, groups = 1, bias = true } = {}) {
      super();
      const kH = Array.isArray(kernelSize) ? kernelSize[0] : kernelSize;
      const kW = Array.isArray(kernelSize) ? kernelSize[1] || kernelSize[0] : kernelSize;
      this.stride = stride;
      this.padding = padding;
      this.outputPadding = outputPadding;
      this.dilation = dilation;
      this.groups = groups;
      this.weight = new Tensor(new Float32Array(inChannels * (outChannels / groups) * kH * kW), [inChannels, outChannels / groups, kH, kW], true);
      this.bias = bias ? Tensor.zeros([outChannels], true) : null;
    }
    forward(x) {
      return x.convTranspose2d(this.weight, this.bias, this.groups, this.stride, this.dilation, this.padding, this.outputPadding);
    }
  }

  function ConvTranspose1d(inChannels, outChannels, kernelSize, opts = {}) {
    const conv = new ConvTranspose2d(inChannels, outChannels, [1, kernelSize], opts);
    const orig = conv.forward.bind(conv);
    conv.forward = (x) => {
      if (x.shape.length !== 3) throw new Error("ConvTranspose1d expects [N,C,W]");
      const [N, C, W] = x.shape;
      const y = orig(x.reshape([N, C, 1, W]));
      return y.reshape([y.shape[0], y.shape[1], y.shape[3]]);
    };
    return conv;
  }

  class GroupNorm extends Module {
    constructor(numGroups, numChannels, { eps = 1e-5, affine = true, requiresGrad = true } = {}) {
      super();
      if (numChannels % numGroups !== 0) throw new Error("GroupNorm channels not divisible by groups");
      this.numGroups = numGroups;
      this.numChannels = numChannels;
      this.eps = eps;
      this.weight = affine ? Tensor.fromArray(new Float32Array(numChannels).fill(1), [numChannels], requiresGrad) : null;
      this.bias = affine ? Tensor.zeros([numChannels], requiresGrad) : null;
    }
    forward(x) {
      const N = x.shape[0];
      const C = x.shape[1];
      const rest = x.numel / (N * C);
      const grouped = x.reshape([N, this.numGroups, (C / this.numGroups) * rest]);
      const y = grouped.layerNorm(this.eps).reshape(x.shape);
      return this.weight ? y.affineChannel(this.weight, this.bias) : y;
    }
  }

  class BatchNorm extends Module {
    constructor(sz, { eps = 1e-5, affine = true, momentum = 0.1 } = {}) {
      super();
      this.eps = eps;
      this.momentum = momentum;
      this.weight = affine ? Tensor.fromArray(new Float32Array(sz).fill(1), [sz], true) : null;
      this.bias = affine ? Tensor.zeros([sz], true) : null;
      this.runningMean = new Float32Array(sz);
      this.runningVar = new Float32Array(sz).fill(1);
    }
    forward(x) {
      if (x.shape.length < 2) throw new Error("BatchNorm expects [N,C,...]");
      const N = x.shape[0], C = x.shape[1];
      const rest = x.numel / (N * C);
      const train = getTraining();
      const mean = new Float32Array(C);
      const va = new Float32Array(C);
      const xd = x.data;
      if (train) {
        for (let c = 0; c < C; c++) {
          let s = 0, cnt = N * rest;
          for (let n = 0; n < N; n++) {
            const off = (n * C + c) * rest;
            for (let i = 0; i < rest; i++) s += xd[off + i];
          }
          mean[c] = s / cnt;
          let v = 0;
          for (let n = 0; n < N; n++) {
            const off = (n * C + c) * rest;
            for (let i = 0; i < rest; i++) {
              const d = xd[off + i] - mean[c];
              v += d * d;
            }
          }
          va[c] = v / cnt;
          this.runningMean[c] = (1 - this.momentum) * this.runningMean[c] + this.momentum * mean[c];
          this.runningVar[c] = (1 - this.momentum) * this.runningVar[c] + this.momentum * va[c];
        }
      } else {
        mean.set(this.runningMean);
        va.set(this.runningVar);
      }
      const out = new Float32Array(x.numel);
      const w = this.weight ? this.weight.data : null;
      const b = this.bias ? this.bias.data : null;
      const inv = new Float32Array(C);
      for (let c = 0; c < C; c++) inv[c] = 1 / Math.sqrt(va[c] + this.eps);
      for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
        const off = (n * C + c) * rest;
        const wc = w ? w[c] : 1, bc = b ? b[c] : 0;
        for (let i = 0; i < rest; i++) out[off + i] = (xd[off + i] - mean[c]) * inv[c] * wc + bc;
      }
      const req = x.requiresGrad || !!(this.weight && this.weight.requiresGrad) || !!(this.bias && this.bias.requiresGrad);
      const parents = [x];
      if (this.weight) parents.push(this.weight);
      if (this.bias) parents.push(this.bias);
      const P = { N, C, rest, hasWeight: this.weight ? 1 : 0, hasBias: this.bias ? 1 : 0 };
      const cpuBackward = (gout) => {
        if (!x.requiresGrad && !(this.weight && this.weight.requiresGrad) && !(this.bias && this.bias.requiresGrad)) return;
        const cnt = N * rest;
        if (x.requiresGrad) {
          if (!x.grad) x.grad = new Float32Array(x.numel);
          for (let c = 0; c < C; c++) {
            let sumG = 0, sumGY = 0;
            const wc = w ? w[c] : 1;
            for (let n = 0; n < N; n++) {
              const off = (n * C + c) * rest;
              for (let i = 0; i < rest; i++) {
                const yhat = (xd[off + i] - mean[c]) * inv[c];
                sumG += gout[off + i] * wc;
                sumGY += gout[off + i] * wc * yhat;
              }
            }
            for (let n = 0; n < N; n++) {
              const off = (n * C + c) * rest;
              for (let i = 0; i < rest; i++) {
                const yhat = (xd[off + i] - mean[c]) * inv[c];
                x.grad[off + i] += inv[c] * (gout[off + i] * wc - sumG / cnt - yhat * sumGY / cnt);
              }
            }
          }
        }
        if (this.weight && this.weight.requiresGrad) {
          if (!this.weight.grad) this.weight.grad = new Float32Array(C);
          for (let c = 0; c < C; c++) {
            let s = 0;
            for (let n = 0; n < N; n++) {
              const off = (n * C + c) * rest;
              for (let i = 0; i < rest; i++) s += gout[off + i] * (xd[off + i] - mean[c]) * inv[c];
            }
            this.weight.grad[c] += s;
          }
        }
        if (this.bias && this.bias.requiresGrad) {
          if (!this.bias.grad) this.bias.grad = new Float32Array(C);
          for (let c = 0; c < C; c++) {
            let s = 0;
            for (let n = 0; n < N; n++) {
              const off = (n * C + c) * rest;
              for (let i = 0; i < rest; i++) s += gout[off + i];
            }
            this.bias.grad[c] += s;
          }
        }
      };
      const backend = gpu();
      if (backend && x.onGPU) {
        if (this.weight) this.weight.toGPU();
        if (this.bias) this.bias.toGPU();
        const gammaBuf = this.weight ? this.weight.gpuBuffer : null;
        const betaBuf = this.bias ? this.bias.gpuBuffer : null;
        let meanBuf, varBuf;
        if (train) {
          const st = backend.batchnormStats(x.gpuBuffer, P);
          meanBuf = st.meanBuf;
          varBuf = st.varBuf;
          // fire-and-forget running-stats update (matches the CPU momentum formula)
          Promise.all([backend.readBuffer(meanBuf, C), backend.readBuffer(varBuf, C)]).then(([m, v]) => {
            for (let c = 0; c < C; c++) {
              this.runningMean[c] = (1 - this.momentum) * this.runningMean[c] + this.momentum * m[c];
              this.runningVar[c] = (1 - this.momentum) * this.runningVar[c] + this.momentum * v[c];
            }
          });
        } else {
          meanBuf = backend.createBufferFromData(Float32Array.from(this.runningMean));
          varBuf = backend.createBufferFromData(Float32Array.from(this.runningVar));
        }
        const outBuf = backend.batchnormFwd(x.gpuBuffer, meanBuf, varBuf, gammaBuf, betaBuf, P);
        return new Tensor(new Float32Array(x.numel), [...x.shape], req, parents, (gout, goutBuf) => {
          if (goutBuf && backend) {
            const st = backend.batchnormBwdStats(x.gpuBuffer, goutBuf, gammaBuf, meanBuf, varBuf, P);
            if (x.requiresGrad) {
              x._pendingGradBuf = backend.batchnormBwdDx(x.gpuBuffer, goutBuf, gammaBuf, meanBuf, varBuf, st.sumGBuf, st.sumGYBuf, P);
            }
            if (this.weight && this.weight.requiresGrad) this.weight._pendingGradBuf = st.dgammaBuf;
            if (this.bias && this.bias.requiresGrad) this.bias._pendingGradBuf = st.dbetaBuf;
            backend.releaseBuffer(st.sumGBuf);
            backend.releaseBuffer(st.sumGYBuf);
          } else {
            cpuBackward(gout);
          }
        }, outBuf, "gpu");
      }
      return new Tensor(out, [...x.shape], req, parents, cpuBackward);
    }
  }

  class LSTMCell extends Module {
    constructor(inputSize, hiddenSize, { bias = true } = {}) {
      super();
      this.inputSize = inputSize;
      this.hiddenSize = hiddenSize;
      this.weightIh = new Tensor(new Float32Array(hiddenSize * 4 * inputSize), [hiddenSize * 4, inputSize], true);
      this.weightHh = new Tensor(new Float32Array(hiddenSize * 4 * hiddenSize), [hiddenSize * 4, hiddenSize], true);
      this.biasIh = bias ? Tensor.zeros([hiddenSize * 4], true) : null;
      this.biasHh = bias ? Tensor.zeros([hiddenSize * 4], true) : null;
    }
    forward(x, hc = null) {
      const B = x.shape[0];
      const H = this.hiddenSize;
      const h = hc ? hc[0] : Tensor.zeros([B, H], false);
      const c = hc ? hc[1] : Tensor.zeros([B, H], false);
      const gi = x.matmul(this.weightIh.transpose2d());
      const gh = h.matmul(this.weightHh.transpose2d());
      let gates = gi.add(gh);
      if (this.biasIh) gates = gates.addBroadcastLast(this.biasIh);
      if (this.biasHh) gates = gates.addBroadcastLast(this.biasHh);
      const parts = gates.splitLast([H, H, H, H]);
      const i = parts[0].sigmoid();
      const f = parts[1].sigmoid();
      const g = parts[2].tanh();
      const o = parts[3].sigmoid();
      const newC = f.mul(c).add(i.mul(g));
      const newH = o.mul(newC.tanh());
      return [newH, newC];
    }
  }

  const nn = {
    Module, Linear, Embedding, RMSNorm, LayerNorm, Dropout,
    Conv2d, Conv1d, ConvTranspose2d, ConvTranspose1d, GroupNorm, BatchNorm, BatchNorm2d: BatchNorm, LSTMCell,
  };
  const optim = { Optimizer, OptimizerGroup, Adam, AdamW, SGD, LAMB, LARS, Muon };

  window.tinygradV0 = {
    Tensor,
    Optimizer,
    OptimizerGroup,
    Module,
    nn,
    optim,
    Adam,
    AdamW,
    SGD,
    LAMB,
    LARS,
    Muon,
    training: false,
    crossEntropy,
    crossEntropyGPU,
    embeddingLookup,
    rmsNorm,
    f32ToF16,
    f16ToF32,
    weightsToF16Base64,
    f16Base64ToWeights,
    encodeInt8Delta,
    decodeInt8Delta,
  };
})();
