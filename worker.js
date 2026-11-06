// All tokenization and inference run off the UI thread.
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';
const MODEL_ID = 'Xenova/gpt2';
let model, tokenizer, stopping, TextStreamer, Tensor, loading, running = false;
const files = new Map();
let lastPercent = 0;
function progress(info) {
 if (!info.file) return;
 const previous = files.get(info.file) || {loaded:0,total:0};
 const total = info.total || previous.total;
 files.set(info.file,{total,loaded:info.status === 'done' ? total : (info.loaded ?? previous.loaded)});
 const values = [...files.values()];
 const loaded = values.reduce((sum,f) => sum+f.loaded,0);
 // An initial estimate avoids a tiny config file showing 100% before weights start.
 const size = Math.max(130_000_000,values.reduce((sum,f) => sum+f.total,0));
 lastPercent = Math.max(lastPercent,Math.min(99,loaded/size*100));
 postMessage({type:'progress',percent:lastPercent,label:`Downloading model · ${(loaded/1e6).toFixed(1)} / ${(size/1e6).toFixed(0)} MB · ${Math.floor(lastPercent)}%`});
}
async function load() {
 if (model) return;
 if (loading) return loading;
 loading = (async () => {
  const lib = await import(TRANSFORMERS_URL);
  const {AutoTokenizer,AutoModelForCausalLM,env,InterruptableStoppingCriteria} = lib;
  ({TextStreamer,Tensor} = lib);
  env.allowLocalModels = false;
  // Pages has no cross-origin isolation headers. Single-thread WASM works without them.
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;
  try {if (!globalThis.caches) env.useBrowserCache = false;else await caches.open('transformers-cache');}
  catch {env.useBrowserCache = false;}
  stopping = new InterruptableStoppingCriteria();
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID,{progress_callback:progress});
  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID,{dtype:'q8',device:'wasm',model_file_name:'decoder_model_merged',progress_callback:progress});
  // WASM can resolve inference promises without yielding to queued worker messages.
  // Give Stop a turn between forward passes (a few ms, not on the UI thread).
  const forward = model.forward.bind(model);
  model.forward = async (...args) => {
   await new Promise(resolve => setTimeout(resolve,0));
   return forward(...args);
  };
  postMessage({type:'progress',percent:100,label:'Download complete · preparing model…'});
 })();
 try {await loading;} finally {loading = null;}
}
self.onmessage = async ({data}) => {
 if (data.type === 'stop') {stopping?.interrupt();return;}
 if (data.type === 'load') {
  try {await load();postMessage({type:'ready'});} catch(error) {report(error);}
  return;
 }
 if (data.type !== 'generate' || running || !model) return;
 running = true;stopping.reset();
 try {
  const {context,...options} = data.settings;
  const encoded = tokenizer(data.text,{add_special_tokens:false});
  const total = encoded.input_ids.dims[1];
  const positions = model.config.n_positions || model.config.max_position_embeddings || 1024;
  const used = Math.min(total,context,positions-options.max_new_tokens);
  if (used < 1) throw new Error('Please enter a longer prompt.');
  // Slice IDs, not decoded text: this preserves the exact latest N tokens.
  const input_ids = encoded.input_ids.slice(null,[total-used,total]);
  const attention_mask = new Tensor('int64',new BigInt64Array(used).fill(1n),[1,used]);
  postMessage({type:'context',total,used});
  const streamer = new TextStreamer(tokenizer,{skip_prompt:true,skip_special_tokens:true,callback_function:text => postMessage({type:'chunk',text})});
  await model.generate({input_ids,attention_mask,...options,do_sample:true,streamer,stopping_criteria:stopping});
  postMessage({type:'done',stopped:stopping.interrupted});
 } catch(error) {report(error);} finally {running = false;}
};
function report(error) {
 console.error(error);
 postMessage({type:'error',loaded:!!model,message:model ? 'Generation ran into a problem. Your text is safe. Try a shorter context or reload the page.' : 'The model could not download or start. Check your internet connection and available memory, then reload. This app needs WebAssembly and access to Hugging Face and jsDelivr.'});
}
