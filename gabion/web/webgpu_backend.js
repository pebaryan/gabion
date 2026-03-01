// webgpu_backend.js
// WebGPU device manager, buffer helpers, shader cache, and kernel dispatch.
// Singleton: call WebGPUBackend.init(gpuDevice) once, then use WebGPUBackend.instance.

(function () {
  "use strict";

  // WGSL kernel sources inlined at build/serve time, or fetched.
  // We store them here after loading.
  const _shaderSources = {};

  // --- Buffer Pool: recycle GPU buffers by size bucket ---
  class BufferPool {
    constructor(device) {
      this.device = device;
      this._free = new Map();  // alignedSize -> [GPUBuffer]
      this._hits = 0;
      this._misses = 0;
    }

    /** Acquire a storage buffer of at least byteSize bytes. */
    acquire(byteSize) {
      // Align to 256 bytes for better reuse across similar-sized allocations
      const key = Math.ceil(byteSize / 256) * 256;
      const list = this._free.get(key);
      if (list && list.length > 0) {
        this._hits++;
        return list.pop();
      }
      this._misses++;
      return this.device.createBuffer({
        size: key,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    /** Return a buffer to the pool for reuse. */
    release(buf) {
      const key = buf.size;
      let list = this._free.get(key);
      if (!list) { list = []; this._free.set(key, list); }
      list.push(buf);
    }

    /** Destroy all pooled buffers and reset. */
    flush() {
      for (const [, list] of this._free) {
        for (const buf of list) buf.destroy();
      }
      this._free.clear();
    }

    get stats() { return { hits: this._hits, misses: this._misses, pooled: [...this._free.values()].reduce((s, l) => s + l.length, 0) }; }
  }

  class WebGPUBackend {
    constructor(device) {
      this.device = device;
      this._pipelineCache = new Map();  // name -> GPUComputePipeline
      this._shaderCache = new Map();    // name -> GPUShaderModule
      this._pool = new BufferPool(device);
      this._activeBatch = null;         // { encoder } when batching
      // Timestamp profiling
      this._profiling = false;
      this._hasTimestamps = device.features.has("timestamp-query");
      this._querySet = null;
      this._queryResolveBuf = null;
      this._queryReadBuf = null;
      this._queryIndex = 0;
      this._queryLabels = [];           // label per dispatch
      this._maxQueries = 128;           // 64 dispatches × 2 timestamps each
    }

    // --- Singleton ---
    static _instance = null;

    static init(device) {
      WebGPUBackend._instance = new WebGPUBackend(device);
      console.log("[webgpu_backend] initialized");
      return WebGPUBackend._instance;
    }

    static get instance() {
      return WebGPUBackend._instance;
    }

    // --- Buffer management ---

    /** Create a GPU storage buffer from Float32Array data. */
    createBufferFromData(data) {
      const buf = this.device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    }

    /** Create an empty GPU storage buffer of given byte size. Uses pool. */
    createEmptyBuffer(byteSize) {
      return this._pool.acquire(byteSize);
    }

    /** Return a buffer to the pool instead of destroying it. */
    releaseBuffer(buf) {
      this._pool.release(buf);
    }

    /** Get pool statistics. */
    get poolStats() { return this._pool.stats; }

    /** Create a uniform buffer from a Uint32Array or Float32Array. */
    createUniformBuffer(data) {
      const buf = this.device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      buf.unmap();
      return buf;
    }

    /** Async readback: GPU buffer -> Float32Array. */
    async readBuffer(gpuBuf, floatCount) {
      const byteSize = floatCount * 4;
      const readBuf = this.device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuf, 0, readBuf, 0, byteSize);
      this.device.queue.submit([encoder.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      readBuf.destroy();
      return result;
    }

    // --- Command batching ---

    /**
     * Begin a command batch. All subsequent kernel dispatches will record
     * into a single CommandEncoder instead of creating their own.
     * Call endBatch() to submit all recorded commands at once.
     */
    beginBatch() {
      if (this._activeBatch) throw new Error("[webgpu_backend] batch already active");
      this._activeBatch = { encoder: this.device.createCommandEncoder(), uniformBufs: [] };
    }

    /** Submit all batched commands and clean up. */
    endBatch() {
      if (!this._activeBatch) throw new Error("[webgpu_backend] no active batch");
      const { encoder, uniformBufs } = this._activeBatch;
      this.device.queue.submit([encoder.finish()]);
      for (const buf of uniformBufs) buf.destroy();
      this._activeBatch = null;
    }

    /** @returns true if currently inside a beginBatch()/endBatch() block. */
    get batching() { return !!this._activeBatch; }

    // --- Profiling ---

    enableProfiling() {
      if (!this._hasTimestamps) {
        console.warn("[webgpu_backend] timestamp-query not supported by this device");
        return false;
      }
      if (this._profiling) return true;
      this._querySet = this.device.createQuerySet({ type: "timestamp", count: this._maxQueries });
      this._queryResolveBuf = this.device.createBuffer({
        size: this._maxQueries * 8,  // 8 bytes per timestamp (u64)
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this._queryReadBuf = this.device.createBuffer({
        size: this._maxQueries * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this._queryIndex = 0;
      this._queryLabels = [];
      this._profiling = true;
      console.log("[webgpu_backend] profiling enabled");
      return true;
    }

    disableProfiling() {
      if (!this._profiling) return;
      this._profiling = false;
      if (this._querySet) { this._querySet.destroy(); this._querySet = null; }
      if (this._queryResolveBuf) { this._queryResolveBuf.destroy(); this._queryResolveBuf = null; }
      if (this._queryReadBuf) { this._queryReadBuf.destroy(); this._queryReadBuf = null; }
      this._queryLabels = [];
      this._queryIndex = 0;
      console.log("[webgpu_backend] profiling disabled");
    }

    /** Reset profiling counters for a new measurement batch. */
    resetProfiling() {
      this._queryIndex = 0;
      this._queryLabels = [];
    }

    /**
     * Resolve timestamps and read back kernel timings.
     * Returns array of { label, durationMs }.
     */
    async readProfilingResults() {
      if (!this._profiling || this._queryIndex === 0) return [];
      const count = this._queryIndex;
      const encoder = this.device.createCommandEncoder();
      encoder.resolveQuerySet(this._querySet, 0, count, this._queryResolveBuf, 0);
      encoder.copyBufferToBuffer(this._queryResolveBuf, 0, this._queryReadBuf, 0, count * 8);
      this.device.queue.submit([encoder.finish()]);
      await this._queryReadBuf.mapAsync(GPUMapMode.READ);
      const times = new BigUint64Array(this._queryReadBuf.getMappedRange());
      const results = [];
      for (let i = 0; i < this._queryLabels.length; i++) {
        const start = times[i * 2];
        const end = times[i * 2 + 1];
        results.push({ label: this._queryLabels[i], durationMs: Number(end - start) / 1e6 });
      }
      this._queryReadBuf.unmap();
      this._queryIndex = 0;
      this._queryLabels = [];
      return results;
    }

    /**
     * Internal: dispatch a compute pass, using the active batch encoder if available.
     * Returns nothing — caller manages output buffer.
     */
    _dispatch(pipeline, bindGroup, workgroupsX, workgroupsY = 1, workgroupsZ = 1, uniformBuf = null) {
      const profiling = this._profiling && this._queryIndex + 2 <= this._maxQueries;
      const tsDesc = profiling ? {
        querySet: this._querySet,
        beginningOfPassWriteIndex: this._queryIndex,
        endOfPassWriteIndex: this._queryIndex + 1,
      } : undefined;

      if (this._activeBatch) {
        const pass = this._activeBatch.encoder.beginComputePass(
          profiling ? { timestampWrites: tsDesc } : undefined
        );
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
        pass.end();
        if (uniformBuf) this._activeBatch.uniformBufs.push(uniformBuf);
      } else {
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass(
          profiling ? { timestampWrites: tsDesc } : undefined
        );
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        if (uniformBuf) uniformBuf.destroy();
      }

      if (profiling) {
        this._queryLabels.push(pipeline._label || "dispatch");
        this._queryIndex += 2;
      }
    }

    // --- Shader / pipeline management ---

    /** Register WGSL source by name (called during init). */
    static registerShader(name, wgslSource) {
      _shaderSources[name] = wgslSource;
    }

    /** Get or compile a shader module. */
    getShaderModule(name) {
      if (this._shaderCache.has(name)) return this._shaderCache.get(name);
      const src = _shaderSources[name];
      if (!src) throw new Error(`[webgpu_backend] shader "${name}" not registered`);
      const mod = this.device.createShaderModule({ code: src });
      this._shaderCache.set(name, mod);
      return mod;
    }

    /** Get or create a compute pipeline for a shader. */
    getPipeline(name, entryPoint = "main") {
      const key = `${name}:${entryPoint}`;
      if (this._pipelineCache.has(key)) return this._pipelineCache.get(key);
      const mod = this.getShaderModule(name);
      const pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: mod, entryPoint },
      });
      pipeline._label = name;  // for profiling
      this._pipelineCache.set(key, pipeline);
      return pipeline;
    }

    // --- Kernel dispatch helpers ---

    /**
     * Dispatch matmul: C[M,N] = A[M,K] * B[K,N]
     * If transposeB=true, B is stored as [N,K] and read transposed.
     * Returns GPUBuffer for C.
     */
    matmul(aBuf, bBuf, M, K, N, transposeB = false) {
      const pipeline = this.getPipeline("matmul");
      const uniformBuf = this.createUniformBuffer(new Uint32Array([M, K, N, transposeB ? 1 : 0]));
      const outBuf = this.createEmptyBuffer(M * N * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: aBuf } },
          { binding: 2, resource: { buffer: bBuf } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(N / 32), Math.ceil(M / 32), 1, uniformBuf);
      return outBuf;
    }

    /**
     * Dispatch elementwise op on buffers.
     * op: 0=add, 1=mul, 2=silu, 3=scale, 4=sub, 5=addScalar
     * For unary ops (silu, scale, addScalar), bBuf can be a dummy 4-byte buffer.
     * Returns GPUBuffer for output.
     */
    elementwise(aBuf, bBuf, len, op, scalar = 0.0) {
      const pipeline = this.getPipeline("elementwise");
      const paramBuf = new ArrayBuffer(16);
      new Uint32Array(paramBuf, 0, 2).set([len, op]);
      new Float32Array(paramBuf, 8, 1).set([scalar]);
      new Uint32Array(paramBuf, 12, 1).set([0]);
      const uniformBuf = this.createUniformBuffer(new Uint8Array(paramBuf));
      const outBuf = this.createEmptyBuffer(len * 4);
      const actualB = bBuf || this.createEmptyBuffer(4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: aBuf } },
          { binding: 2, resource: { buffer: actualB } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(len / 256), 1, 1, uniformBuf);
      if (!bBuf) actualB.destroy();
      return outBuf;
    }

    /**
     * Dispatch row-wise reduction.
     * op: 0=sum, 1=max, 2=sumSquares
     * Input: [rows, cols] buffer, Output: [rows] buffer.
     */
    reduce(inputBuf, rows, cols, op) {
      const pipeline = this.getPipeline("reduce");
      const uniformBuf = this.createUniformBuffer(new Uint32Array([rows, cols, op, 0]));
      const outBuf = this.createEmptyBuffer(rows * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: inputBuf } },
          { binding: 2, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, rows, 1, 1, uniformBuf);
      return outBuf;
    }

    /** Copy one GPU buffer into another (same size). */
    copyBuffer(srcBuf, dstBuf, byteSize) {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(srcBuf, 0, dstBuf, 0, byteSize);
      this.device.queue.submit([encoder.finish()]);
    }

    /** Upload Float32Array data into an existing GPU buffer. */
    writeBuffer(gpuBuf, data) {
      this.device.queue.writeBuffer(gpuBuf, 0, data);
    }

    /**
     * Row-wise softmax with optional causal mask.
     * Input: [rows, cols] buffer. Output: [rows, cols] buffer.
     * If maskFlag=1, applies causal mask (col > row%cols → -inf) before softmax.
     */
    softmax(inputBuf, rows, cols, maskFlag = 0) {
      const pipeline = this.getPipeline("softmax");
      const uniformBuf = this.createUniformBuffer(new Uint32Array([rows, cols, maskFlag, 0]));
      const outBuf = this.createEmptyBuffer(rows * cols * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: inputBuf } },
          { binding: 2, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, rows, 1, 1, uniformBuf);
      return outBuf;
    }

    /**
     * Rotary Position Embedding on GPU.
     * x: [BH * T * headDim] buffer
     * freqCosBuf, freqSinBuf: [T * headDim] buffers (precomputed cos/sin tables)
     * Returns output buffer [BH * T * headDim].
     */
    rope(xBuf, freqCosBuf, freqSinBuf, BH, T, headDim) {
      const pipeline = this.getPipeline("rope");
      const halfDim = headDim >>> 1;
      const uniformBuf = this.createUniformBuffer(new Uint32Array([BH, T, headDim, halfDim]));
      const outBuf = this.createEmptyBuffer(BH * T * headDim * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: xBuf } },
          { binding: 2, resource: { buffer: freqCosBuf } },
          { binding: 3, resource: { buffer: freqSinBuf } },
          { binding: 4, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(BH * T * halfDim / 256), 1, 1, uniformBuf);
      return outBuf;
    }

    /**
     * Batched matmul: C[b, M, N] = A[b, M, K] * B[b, K, N]  for b in [0, B)
     * A: [B*M*K], B: [B*K*N], C: [B*M*N] — contiguous batch slices.
     * Single kernel dispatch with batch dimension in workgroup z-axis.
     * If transposeB=true, B is stored as [B*N*K] and read transposed.
     */
    batchedMatmul(aBuf, bBuf, B, M, K, N, transposeB = false) {
      const pipeline = this.getPipeline("batched_matmul");
      const uniformBuf = this.createUniformBuffer(new Uint32Array([M, K, N, transposeB ? 1 : 0]));
      const outBuf = this.createEmptyBuffer(B * M * N * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: aBuf } },
          { binding: 2, resource: { buffer: bBuf } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(N / 16), Math.ceil(M / 16), B, uniformBuf);
      return outBuf;
    }

    /**
     * Fused attention: attn[BH, T, T] = softmax(causalMask(Q @ K^T * scale))
     * Q: [BH * T * headDim], K: [BH * T * headDim] buffers.
     * Combines dot-product, scaling, causal masking, and softmax in one kernel.
     * Constraint: T <= 256.
     * Returns GPUBuffer for attn weights [BH * T * T].
     */
    fusedAttention(qBuf, kBuf, BH, T, headDim) {
      const pipeline = this.getPipeline("fused_attention");
      const scale = 1.0 / Math.sqrt(headDim);
      const paramBuf = new ArrayBuffer(16);
      new Uint32Array(paramBuf, 0, 3).set([BH, T, headDim]);
      new Float32Array(paramBuf, 12, 1).set([scale]);
      const uniformBuf = this.createUniformBuffer(new Uint8Array(paramBuf));
      const outBuf = this.createEmptyBuffer(BH * T * T * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: qBuf } },
          { binding: 2, resource: { buffer: kBuf } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, BH * T, 1, 1, uniformBuf);
      return outBuf;
    }

    /**
     * Softmax backward: dScores = attn * (dAttn - dot(dAttn, attn)) * scale
     * with causal mask zeroing (positions c > row % cols set to 0).
     * attnBuf, dAttnBuf: [rows * cols] buffers.
     * Returns GPUBuffer for dScores [rows * cols].
     */
    softmaxBackward(attnBuf, dAttnBuf, rows, cols, scale) {
      const pipeline = this.getPipeline("softmax_backward");
      const paramBuf = new ArrayBuffer(16);
      new Uint32Array(paramBuf, 0, 2).set([rows, cols]);
      new Float32Array(paramBuf, 8, 1).set([scale]);
      new Uint32Array(paramBuf, 12, 1).set([0]);
      const uniformBuf = this.createUniformBuffer(new Uint8Array(paramBuf));
      const outBuf = this.createEmptyBuffer(rows * cols * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: attnBuf } },
          { binding: 2, resource: { buffer: dAttnBuf } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, rows, 1, 1, uniformBuf);
      return outBuf;
    }

    /**
     * RoPE backward: computes gradient w.r.t. RoPE input.
     * dOutBuf: [BH * T * headDim] upstream gradient buffer.
     * freqCosBuf, freqSinBuf: [T * headDim] precomputed cos/sin tables.
     * Returns GPUBuffer for dX [BH * T * headDim].
     */
    ropeBackward(dOutBuf, freqCosBuf, freqSinBuf, BH, T, headDim) {
      const pipeline = this.getPipeline("rope_backward");
      const halfDim = headDim >>> 1;
      const uniformBuf = this.createUniformBuffer(new Uint32Array([BH, T, headDim, halfDim]));
      const outBuf = this.createEmptyBuffer(BH * T * headDim * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: dOutBuf } },
          { binding: 2, resource: { buffer: freqCosBuf } },
          { binding: 3, resource: { buffer: freqSinBuf } },
          { binding: 4, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(BH * T * halfDim / 256), 1, 1, uniformBuf);
      return outBuf;
    }

    /**
     * Batched transpose: out[b, j, i] = in[b, i, j]
     * Input: [B * M * N], Output: [B * N * M].
     */
    batchedTranspose(inputBuf, B, M, N) {
      const pipeline = this.getPipeline("batched_transpose");
      const uniformBuf = this.createUniformBuffer(new Uint32Array([M, N, 0, 0]));
      const outBuf = this.createEmptyBuffer(B * M * N * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: inputBuf } },
          { binding: 2, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(N / 16), Math.ceil(M / 16), B, uniformBuf);
      return outBuf;
    }

    /**
     * RMSNorm backward: computes dX and dW from upstream gradient.
     * xBuf: [rows * d] input, goutBuf: [rows * d] upstream grad,
     * wBuf: [d] weight (or null), eps: epsilon.
     * Returns { dXBuf: [rows*d], dWBuf: [rows*d] per-row contributions }.
     * Caller must reduce dWBuf across rows to get final [d] dW.
     */
    rmsNormBackward(xBuf, goutBuf, wBuf, rows, d, eps) {
      const pipeline = this.getPipeline("rmsnorm_backward");
      const hasWeight = wBuf ? 1 : 0;
      const paramBuf = new ArrayBuffer(16);
      new Uint32Array(paramBuf, 0, 2).set([rows, d]);
      new Float32Array(paramBuf, 8, 1).set([eps]);
      new Uint32Array(paramBuf, 12, 1).set([hasWeight]);
      const uniformBuf = this.createUniformBuffer(new Uint8Array(paramBuf));
      const dXBuf = this.createEmptyBuffer(rows * d * 4);
      const dWBuf = this.createEmptyBuffer(rows * d * 4);
      const actualW = wBuf || this.createEmptyBuffer(d * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: xBuf } },
          { binding: 2, resource: { buffer: goutBuf } },
          { binding: 3, resource: { buffer: actualW } },
          { binding: 4, resource: { buffer: dXBuf } },
          { binding: 5, resource: { buffer: dWBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, rows, 1, 1, uniformBuf);
      if (!wBuf) actualW.destroy();
      return { dXBuf, dWBuf };
    }

    adamUpdate(gradBuf, mBuf, vBuf, paramBuf, len, beta1, beta2, effLr, bc1, bc2, eps) {
      const pipeline = this.getPipeline("adam_update");
      const ab = new ArrayBuffer(32);
      new Uint32Array(ab, 0, 1).set([len]);
      // offset 4: _pad (leave as 0)
      new Float32Array(ab, 8, 6).set([beta1, beta2, effLr, bc1, bc2, eps]);
      const uniformBuf = this.createUniformBuffer(new Uint8Array(ab));
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: gradBuf } },
          { binding: 2, resource: { buffer: mBuf } },
          { binding: 3, resource: { buffer: vBuf } },
          { binding: 4, resource: { buffer: paramBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(len / 256), 1, 1, uniformBuf);
    }

    sgdUpdate(gradBuf, paramBuf, len, lr) {
      const pipeline = this.getPipeline("sgd_update");
      const ab = new ArrayBuffer(16);
      new Uint32Array(ab, 0, 1).set([len]);
      new Float32Array(ab, 8, 1).set([lr]);
      const uniformBuf = this.createUniformBuffer(new Uint8Array(ab));
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: gradBuf } },
          { binding: 2, resource: { buffer: paramBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(len / 256), 1, 1, uniformBuf);
    }

    crossEntropyForward(logitsBuf, targetsBuf, N, V) {
      const pipeline = this.getPipeline("cross_entropy_forward");
      const uniformBuf = this.createUniformBuffer(new Uint32Array([N, V, 0, 0]));
      const probsBuf = this.createEmptyBuffer(N * V * 4);
      const lossesBuf = this.createEmptyBuffer(N * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: logitsBuf } },
          { binding: 2, resource: { buffer: targetsBuf } },
          { binding: 3, resource: { buffer: probsBuf } },
          { binding: 4, resource: { buffer: lossesBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, N, 1, 1, uniformBuf);
      return { probsBuf, lossesBuf };
    }

    crossEntropyBackward(probsBuf, targetsBuf, N, V, scale) {
      const pipeline = this.getPipeline("cross_entropy_backward");
      const ab = new ArrayBuffer(16);
      new Uint32Array(ab, 0, 2).set([N, V]);
      new Float32Array(ab, 8, 1).set([scale]);
      const uniformBuf = this.createUniformBuffer(new Uint8Array(ab));
      const dLogitsBuf = this.createEmptyBuffer(N * V * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: probsBuf } },
          { binding: 2, resource: { buffer: targetsBuf } },
          { binding: 3, resource: { buffer: dLogitsBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(N * V / 256), 1, 1, uniformBuf);
      return dLogitsBuf;
    }

    embeddingForward(embBuf, indicesBuf, BT, D, V) {
      const pipeline = this.getPipeline("embedding_forward");
      const uniformBuf = this.createUniformBuffer(new Uint32Array([BT, D, V, 0]));
      const outBuf = this.createEmptyBuffer(BT * D * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: embBuf } },
          { binding: 2, resource: { buffer: indicesBuf } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(BT * D / 256), 1, 1, uniformBuf);
      return outBuf;
    }

    siluMul(aBuf, N, dFF) {
      const pipeline = this.getPipeline("silu_mul");
      const len = N * dFF;
      const uniformBuf = this.createUniformBuffer(new Uint32Array([len, dFF, 0, 0]));
      const outBuf = this.createEmptyBuffer(len * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: aBuf } },
          { binding: 2, resource: { buffer: outBuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(len / 256), 1, 1, uniformBuf);
      return outBuf;
    }

    siluMulBackward(dOutBuf, aBuf, N, dFF) {
      const pipeline = this.getPipeline("silu_mul_backward");
      const len = N * dFF;
      const uniformBuf = this.createUniformBuffer(new Uint32Array([len, dFF, 0, 0]));
      const dABuf = this.createEmptyBuffer(N * 2 * dFF * 4);
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: dOutBuf } },
          { binding: 2, resource: { buffer: aBuf } },
          { binding: 3, resource: { buffer: dABuf } },
        ],
      });
      this._dispatch(pipeline, bindGroup, Math.ceil(len / 256), 1, 1, uniformBuf);
      return dABuf;
    }
  }

  window.WebGPUBackend = WebGPUBackend;
})();
