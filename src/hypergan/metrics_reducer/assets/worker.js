import {Reducer, appendFrame, mergeBootstraps, validateBootstrap} from './host.js';
const loaded = Reducer.load();
self.onmessage = async ({data}) => {
  const {id, op} = data;
  try {
    const reducer = await loaded;
    let value;
    if (op === 'request') value = reducer.request(data.request);
    else if (op === 'continue') {
      let view = data.bootstrap;
      validateBootstrap(reducer, view);
      if (!Array.isArray(data.frames) || data.frames.length > 1024) throw new Error('Worker frame batch exceeds 1024');
      for (const frame of data.frames) view = appendFrame(reducer, view, frame);
      value = {bootstrap:view, value:reducer.finalize(view.state), module_sha256:reducer.spec.sha256,
               memory_bytes:reducer.exports.memory.buffer.byteLength};
    } else if (op === 'merge') value = mergeBootstraps(reducer, data.left, data.right);
    else throw new Error('Unknown worker operation');
    self.postMessage({id,ok:value});
  } catch (error) { self.postMessage({id,error:String(error.message || error)}); }
};
