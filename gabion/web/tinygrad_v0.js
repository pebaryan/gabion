// tinygrad_v0.js
// Minimal browser-side autograd runtime for federated local training prototype.
// Supports optional WebGPU acceleration via WebGPUBackend (GPU path with CPU fallback).
(function () {
  "use strict";

  // --- GPU helper: check if backend is available ---
  function gpu() {
    return window.WebGPUBackend && WebGPUBackend.instance;
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

    /** Release GPU buffer. */
    releaseGPU() {
      if (this.gpuBuffer) {
        this.gpuBuffer.destroy();
        this.gpuBuffer = null;
      }
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
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = this.data[i] + other.data[i];
      const req = this.requiresGrad || other.requiresGrad;
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

    /** Scale by scalar. */
    scale(s) {
      const n = this.numel;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = this.data[i] * s;
      return new Tensor(out, [...this.shape], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(n);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i] * s;
      });
    }

    /** Reshape (view, no copy). Product of dims must match numel. */
    reshape(newShape) {
      const n = newShape.reduce((a, b) => a * b, 1);
      if (n !== this.numel) throw new Error(`reshape: ${this.shape} -> ${newShape} size mismatch`);
      const origShape = this.shape;
      return new Tensor(this.data, newShape, this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(this.data.length);
        for (let i = 0; i < n; i++) this.grad[i] += gout[i];
      });
    }

    /** Softmax along last axis. Input shape [..., C]. */
    softmax() {
      const shape = this.shape;
      const C = shape[shape.length - 1];
      const rows = this.numel / C;
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
          for (let j = 0; j < C; j++) {
            this.grad[off + j] += outData[off + j] * (gout[off + j] - dot);
          }
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
      const out = new Float32Array(B * m * n);
      const aData = this.data;
      const bData = other.data;

      for (let b = 0; b < B; b++) {
        const aOff = b * m * k1;
        const bOff = b * k2 * n;
        const oOff = b * m * n;
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            let s = 0.0;
            for (let k = 0; k < k1; k++) {
              s += aData[aOff + i * k1 + k] * bData[bOff + k * n + j];
            }
            out[oOff + i * n + j] = s;
          }
        }
      }

      const req = this.requiresGrad || other.requiresGrad;
      return new Tensor(out, [B, m, n], req, [this, other], (gout) => {
        // dA[b] = gout[b] @ B[b]^T
        if (this.requiresGrad) {
          if (!this.grad) this.grad = new Float32Array(this.data.length);
          for (let b = 0; b < B; b++) {
            const gOff = b * m * n;
            const bOff = b * k2 * n;
            const aOff = b * m * k1;
            for (let i = 0; i < m; i++) {
              for (let k = 0; k < k1; k++) {
                let s = 0.0;
                for (let j = 0; j < n; j++) s += gout[gOff + i * n + j] * bData[bOff + k * n + j];
                this.grad[aOff + i * k1 + k] += s;
              }
            }
          }
        }
        // dB[b] = A[b]^T @ gout[b]
        if (other.requiresGrad) {
          if (!other.grad) other.grad = new Float32Array(other.data.length);
          for (let b = 0; b < B; b++) {
            const aOff = b * m * k1;
            const gOff = b * m * n;
            const bOff = b * k2 * n;
            for (let k = 0; k < k1; k++) {
              for (let j = 0; j < n; j++) {
                let s = 0.0;
                for (let i = 0; i < m; i++) s += aData[aOff + i * k1 + k] * gout[gOff + i * n + j];
                other.grad[bOff + k * n + j] += s;
              }
            }
          }
        }
      });
    }

    /**
     * Transpose last two dims of a 3D tensor: [B, M, N] -> [B, N, M].
     */
    transpose3d() {
      if (this.shape.length !== 3) throw new Error(`transpose3d requires 3D, got ${this.shape}`);
      const [B, m, n] = this.shape;
      const out = new Float32Array(B * m * n);
      for (let b = 0; b < B; b++) {
        const off = b * m * n;
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            out[b * n * m + j * m + i] = this.data[off + i * n + j];
          }
        }
      }
      return new Tensor(out, [B, n, m], this.requiresGrad, [this], (gout) => {
        if (!this.requiresGrad) return;
        if (!this.grad) this.grad = new Float32Array(this.data.length);
        for (let b = 0; b < B; b++) {
          const off = b * n * m;
          for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
              this.grad[b * m * n + i * n + j] += gout[off + j * m + i];
            }
          }
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

  // --- Sync CPU training (original path, kept as fallback) ---
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
      return {
        updated: out,
        loss: Number(lastLoss),
        sampleCount: samples,
        mode: "v0-bbt-embed",
      };
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

  // --- Async GPU-accelerated training ---
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
    if (!backend) {
      // No GPU: fall back to CPU path
      return trainLocalV0(weights, opts);
    }

    const V = opts.vocabSize || 256;
    const D = opts.dModel || 64;
    const embBlock = V * D;

    if (weights.length < embBlock) {
      // Too few weights for embedding path — use CPU
      return trainLocalV0(weights, opts);
    }

    const lr = Math.max(1e-7, opts.lr || 5e-4);
    const epochs = Math.max(1, opts.epochs || 1);
    const batchSize = Math.max(8, opts.batchSize || 64);
    let seed = opts.seed >>> 0;

    let lastLoss = 0.0;
    let samples = 0;

    const eView = weights.subarray(0, embBlock);
    const E = Tensor.fromArray(eView, [V, D], true);
    E.toGPU();  // Upload embedding to GPU
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

      // Forward: embedding lookup (CPU scatter) -> GPU matmul for projection
      const h = embeddingLookup(E, batch.x);
      const hn = rmsNorm(h, 1e-6);
      const Et = E.transpose2d();
      const logits = hn.matmul(Et);

      // Cross-entropy needs CPU logits
      if (logits.onGPU) await logits.toCPU();
      const loss = crossEntropy(logits, batch.y);

      // Backward (CPU path for now — GPU backward is wired but we resolve on CPU)
      loss.backward();

      // Resolve any pending GPU grad buffers
      await loss.resolveGrads();

      const g = E.grad || new Float32Array(embBlock);
      for (let i = 0; i < embBlock; i++) E.data[i] -= lr * g[i];
      E.markCPUDirty();

      // Re-upload updated embedding for next epoch
      if (e < epochs - 1) {
        E.toGPU();
        if (opts.debugSync) await debugAssertGPUHead(E, `E:epoch${e + 1}`);
      }

      lastLoss = loss.data[0];
      samples += bsz;
    }

    // Clean up GPU buffers
    E.releaseGPU();

    const out = new Float32Array(weights.length);
    out.set(weights);
    out.set(E.data, 0);
    return {
      updated: out,
      loss: Number(lastLoss),
      sampleCount: samples,
      mode: "v0-bbt-embed-webgpu",
    };
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

  window.tinygradV0 = {
    Tensor,
    crossEntropy,
    crossEntropyGPU,
    embeddingLookup,
    rmsNorm,
    sampleBatch,
    trainLocalV0,
    trainLocalV0Async,
    debugAssertGPUHead,
    f32ToF16,
    f16ToF32,
    weightsToF16Base64,
    f16Base64ToWeights,
    encodeInt8Delta,
    decodeInt8Delta,
  };
})();
