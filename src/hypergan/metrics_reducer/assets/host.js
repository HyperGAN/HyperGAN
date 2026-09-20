/* Thin transport host; all reducer mathematics runs in the bundled core WASM. */
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', {fatal: true});
export class ReducerError extends Error {}
function finiteJSON(value, depth = 0) {
  if (depth > 64) throw new ReducerError('JSON nesting exceeds 64');
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number' && Number.isFinite(value)) return;
  if (Array.isArray(value)) { value.forEach(v => finiteJSON(v, depth + 1)); return; }
  if (value && Object.getPrototypeOf(value) === Object.prototype) {
    Object.values(value).forEach(v => finiteJSON(v, depth + 1)); return;
  }
  throw new ReducerError('Request must contain finite JSON values');
}
/* Web Crypto exists only in secure contexts (HTTPS, localhost). A viewer opened
   over plain HTTP from another host must still verify the bundled module, so this
   portable SHA-256 replaces crypto.subtle there. Both paths reject a mismatch. */
const SHA256_K = Uint32Array.of(
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2);
export function sha256Hex(data) {
  const input = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
  const padded = new Uint8Array(((input.length + 9 + 63) >>> 6) << 6);
  padded.set(input); padded[input.length] = 0x80;
  const block = new DataView(padded.buffer);
  block.setUint32(padded.length - 8, Math.floor(input.length / 0x20000000));
  block.setUint32(padded.length - 4, (input.length % 0x20000000) * 8);
  const w = new Uint32Array(64);
  let h0=0x6a09e667,h1=0xbb67ae85,h2=0x3c6ef372,h3=0xa54ff53a;
  let h4=0x510e527f,h5=0x9b05688c,h6=0x1f83d9ab,h7=0x5be0cd19;
  for (let at = 0; at < padded.length; at += 64) {
    for (let i = 0; i < 16; i++) w[i] = block.getUint32(at + i * 4);
    for (let i = 16; i < 64; i++) {
      const x = w[i-15], y = w[i-2];
      const s0 = ((x>>>7)|(x<<25)) ^ ((x>>>18)|(x<<14)) ^ (x>>>3);
      const s1 = ((y>>>17)|(y<<15)) ^ ((y>>>19)|(y<<13)) ^ (y>>>10);
      w[i] = (w[i-16] + s0 + w[i-7] + s1) >>> 0;
    }
    let a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
    for (let i = 0; i < 64; i++) {
      const S1 = ((e>>>6)|(e<<26)) ^ ((e>>>11)|(e<<21)) ^ ((e>>>25)|(e<<7));
      const t1 = (h + S1 + ((e & f) ^ (~e & g)) + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a>>>2)|(a<<30)) ^ ((a>>>13)|(a<<19)) ^ ((a>>>22)|(a<<10));
      const t2 = (S0 + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      h=g; g=f; f=e; e=(d+t1)>>>0; d=c; c=b; b=a; a=(t1+t2)>>>0;
    }
    h0=(h0+a)>>>0; h1=(h1+b)>>>0; h2=(h2+c)>>>0; h3=(h3+d)>>>0;
    h4=(h4+e)>>>0; h5=(h5+f)>>>0; h6=(h6+g)>>>0; h7=(h7+h)>>>0;
  }
  return [h0,h1,h2,h3,h4,h5,h6,h7].map(v => v.toString(16).padStart(8, '0')).join('');
}
export async function moduleDigest(binary) {
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    try {
      return [...new Uint8Array(await subtle.digest('SHA-256', binary))]
        .map(v => v.toString(16).padStart(2, '0')).join('');
    } catch (error) { /* Web Crypto refused; the portable digest below is authoritative. */ }
  }
  try { return sha256Hex(binary); }
  catch (error) {
    throw new ReducerError(`Cannot compute the bundled reducer digest: ${error.message || error}`);
  }
}
export class Reducer {
  static async load(base = new URL('./', import.meta.url)) {
    const [specResponse, binaryResponse] = await Promise.all([
      fetch(new URL('reducer.json', base)), fetch(new URL('reducer.wasm', base))]);
    if (!specResponse.ok || !binaryResponse.ok) throw new ReducerError('Cannot load bundled reducer');
    const spec = await specResponse.json();
    const binary = await binaryResponse.arrayBuffer();
    if (await moduleDigest(binary) !== spec.sha256) throw new ReducerError('Bundled reducer module digest mismatch');
    const module = await WebAssembly.compile(binary);
    if (WebAssembly.Module.imports(module).length) throw new ReducerError('Reducer must have no host imports');
    const instance = await WebAssembly.instantiate(module, {});
    if (instance.exports.abi_version() !== 1 || spec.abi !== 1) throw new ReducerError('Unsupported reducer ABI');
    return new Reducer(instance.exports, spec);
  }
  constructor(exports, spec) { this.exports = exports; this.spec = spec; this.failed = false; }
  request(request) {
    if (this.failed) throw new ReducerError('Reducer instance trapped; create a new instance');
    finiteJSON(request);
    const encoded = encoder.encode(JSON.stringify(request));
    if (encoded.length > 262144) throw new ReducerError('Request exceeds 262144 bytes');
    let result;
    try {
      new Uint8Array(this.exports.memory.buffer, this.exports.input_ptr(), encoded.length).set(encoded);
      const length = this.exports.execute(encoded.length);
      if (!(length > 0 && length <= 65536)) throw new ReducerError('Invalid reducer response length');
      result = JSON.parse(decoder.decode(new Uint8Array(this.exports.memory.buffer, this.exports.output_ptr(), length)));
    } catch (error) {
      if (error instanceof WebAssembly.RuntimeError) this.failed = true;
      throw error;
    }
    if ('error' in result) throw new ReducerError(result.error);
    return result.ok;
  }
  identity(reducer) { return this.request({op:'identity', reducer}); }
  add(state, values) { return this.request({op:'add', state, values}); }
  merge(left, right) { return this.request({op:'merge', left, right}); }
  finalize(state) { return this.request({op:'finalize', state}); }
}
function offset(n) {
  if (!Number.isSafeInteger(n) || n < 0) throw new ReducerError('Coverage offset must be a nonnegative safe integer');
}
export function validateBootstrap(reducer, view) {
  const keys = ['identity','reducer','module_sha256','start','end','state'];
  if (!view || Object.keys(view).length !== keys.length || keys.some(k => !(k in view))) throw new ReducerError('Invalid bootstrap fields');
  offset(view.start); offset(view.end);
  if (view.start > view.end) throw new ReducerError('Reversed coverage range');
  if (typeof view.identity !== 'string' || view.identity.length < 1 || view.identity.length > 256 || !/^[\x00-\x7F]+$/.test(view.identity)) throw new ReducerError('Invalid bootstrap identity');
  if (view.module_sha256 !== reducer.spec.sha256) throw new ReducerError('Bootstrap module digest mismatch');
  if (!view.state || view.state.reducer !== view.reducer) throw new ReducerError('Bootstrap reducer mismatch');
  reducer.finalize(view.state);
  if (view.start === view.end && JSON.stringify(view.state) !== JSON.stringify(reducer.identity(view.reducer))) {
    // Compare fields without making JSON object insertion order part of state identity.
    const identity = reducer.identity(view.reducer);
    if (Object.keys(identity).some(k => JSON.stringify(identity[k]) !== JSON.stringify(view.state[k]))) throw new ReducerError('Empty coverage must contain identity state');
  }
}
export function appendFrame(reducer, view, frame) {
  validateBootstrap(reducer, view);
  const {identity,start,end,values} = frame;
  offset(start); offset(end);
  if (identity !== view.identity) throw new ReducerError('Frame identity mismatch');
  if (start >= end) throw new ReducerError('Frame must advance coverage');
  if (start >= view.start && end <= view.end) return view;
  if (start !== view.end) throw new ReducerError('Coverage gap or partial overlap; request a new bootstrap');
  return {...view, end, state: reducer.add(view.state, values)};
}
export function mergeBootstraps(reducer, left, right) {
  validateBootstrap(reducer,left); validateBootstrap(reducer,right);
  if (['identity','reducer','module_sha256'].some(k => left[k] !== right[k])) throw new ReducerError('Cannot merge incompatible view identities');
  if (left.end !== right.start) throw new ReducerError('Only adjacent disjoint coverage can merge');
  return {...left, end:right.end, state:reducer.merge(left.state,right.state)};
}
