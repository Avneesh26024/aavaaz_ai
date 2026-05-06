/**
 * pcm-processor.js
 *
 * AudioWorklet processor that captures float32 microphone samples,
 * converts them to Int16 PCM, and posts them back to the main thread.
 *
 * Placed in /public/ so it is served as a static asset and can be loaded
 * via audioContext.audioWorklet.addModule('/pcm-processor.js').
 */

class PcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Buffer accumulates samples across process() calls before posting
    this._buffer = [];
    // Post every ~128ms worth of samples at 16 kHz = 2048 samples
    this._chunkSize = 2048;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array, mono channel

    for (let i = 0; i < samples.length; i++) {
      // Clamp float32 to [-1, 1] then scale to Int16 range
      const clamped = Math.max(-1, Math.min(1, samples[i]));
      const int16 = clamped < 0 ? clamped * 32768 : clamped * 32767;
      this._buffer.push(Math.round(int16));
    }

    if (this._buffer.length >= this._chunkSize) {
      const chunk = new Int16Array(this._buffer.splice(0, this._chunkSize));
      this.port.postMessage(chunk, [chunk.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PcmProcessor);
