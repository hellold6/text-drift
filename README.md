# Text Drift

A tiny browser-only GPT-2 text continuation playground. No backend, framework, build step, API key, chat template, or system prompt. Prompts and generated text stay in your browser. A network connection is needed for the app, CDN runtime, and first model download.

## Files

- `index.html`: the complete UI, inline styles, and UI logic. Change `const APP_NAME = "Text Drift";` near the top of its script to rename the header and browser title.
- `worker.js`: module worker handling model download, tokenization, sampling, and streaming.
- `README.md`: this guide. Only the first two files are required to run.
- `shots/`: optional test screenshots, not app dependencies.

## Run locally

Serve the folder, rather than opening a `file://` URL:

```sh
python3 -m http.server 8000
```

Open <http://localhost:8000>. Use a modern browser with WebAssembly. No WebGPU or cross-origin isolation headers are required. The WASM runtime is single-threaded inside a Web Worker, so inference does not block the UI.

The first visit downloads about 130 MB of quantized GPT-2 weights/tokenizer files plus the runtime. The progress bar aggregates model files by bytes. Browser Cache Storage normally reuses these files on later visits. Cache access and settings storage are guarded so disabled/full browser storage is non-fatal, but caching is not guaranteed in private mode. This is not a fully offline-installable PWA: reloading still requires the page and CDN runtime to be available.

## Use

Write or paste a prompt, then Generate or Ctrl/Cmd+Enter. Continuation streams into the same editor in a contrasting colour. Click again to extend it. User typing after generation is normal colour. Editing inside a generated run changes that run back to normal colour. Paste inserts plain text only. Stop cancels at the next inference boundary. A worker-termination fallback after 1.8 seconds handles WASM backends that do not yield promptly, then reloads the cached model. Already-streamed text is preserved.

Settings are collapsible and persisted locally. Top-k 0 disables that filter. Length is a maximum: the model may stop early at its end-of-text token. Only the latest N prompt tokens are fed to the model, without deleting old editor text. GPT-2 has 1,024 total positions, including newly generated tokens, so the effective context is `min(prompt tokens, context slider, 1024 - length)`. For example, context 1024 and length 500 use the latest 524 prompt tokens. The status line displays the actual number used.

Model text is not fact-checked or moderated and may be inaccurate, biased, or offensive.

## Deploy on GitHub Pages

1. Create a public repository, such as `text-drift`.
2. Commit `index.html`, `worker.js`, and optionally this README to its `main` branch at the repository root.
3. Open **Settings → Pages → Build and deployment**.
4. Select **Deploy from a branch**, **main**, and **/ (root)**, then Save.
5. Wait for the Pages deployment to complete. Open `https://YOUR-USERNAME.github.io/text-drift/`.
6. Verify the progress bar, model ready status, and generated text in your browser. Make sure both `index.html` and `worker.js` are served from the same folder.

The relative worker URL also works under a Pages project subpath. No secrets or server environment variables are needed.

## Runtime and model choices

Pinned Transformers.js **3.8.1**:

<https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js>

Default model: <https://huggingface.co/Xenova/gpt2>, `dtype: 'q8'`. This repository has a legacy ONNX filename, so `model_file_name: 'decoder_model_merged'` is required with Transformers.js 3.x. It selects `onnx/decoder_model_merged_quantized.onnx` (128,315,766 bytes). Do not remove the filename override for this model.

To swap models, change `MODEL_ID` in `worker.js`, verify the repository has a supported causal-LM ONNX export, and adjust the filename, dtype, position limit, and displayed download estimate as needed. Keep the base model and plain text input, not a chat/instruction model.

- **Smaller:** <https://huggingface.co/Xenova/distilgpt2> is a verified public ONNX repository. It includes both `model_quantized.onnx` and `decoder_model_merged_quantized.onnx`, so the current filename override can stay. It uses fewer layers, needs less memory, and produces lower-quality continuations.
- **Larger:** GPT-2 medium has about 355 million parameters. The base model is available at <https://huggingface.co/openai-community/gpt2-medium>, but that is not a drop-in browser ONNX repository. `Xenova/gpt2-medium` was not publicly accessible during verification, so do not assume that ID works. Find or export a Transformers.js-compatible ONNX version first. An 8-bit model is roughly 350–400 MB or more with export overhead and needs significantly more runtime RAM, especially on phones.

Changing model IDs creates a separate browser cache entry. You can remove cached weights through your browser's site-data controls.
