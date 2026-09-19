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
export class Reducer {
  static async load(base = new URL('./', import.meta.url)) {
    const [specResponse, binaryResponse] = await Promise.all([
      fetch(new URL('reducer.json', base)), fetch(new URL('reducer.wasm', base))]);
    if (!specResponse.ok || !binaryResponse.ok) throw new ReducerError('Cannot load bundled reducer');
    const spec = await specResponse.json();
    const binary = await binaryResponse.arrayBuffer();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', binary))]
      .map(v => v.toString(16).padStart(2, '0')).join('');
    if (digest !== spec.sha256) throw new ReducerError('Bundled reducer module digest mismatch');
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
