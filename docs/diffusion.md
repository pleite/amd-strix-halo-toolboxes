# Diffusion LLMs on Strix Halo (Gemma Diffusion)

This toolbox runs **diffusion language models** (DLLMs) with `llama.cpp`'s
`llama-diffusion-cli` on AMD "Strix Halo" iGPUs via ROCm. Unlike autoregressive
models, diffusion LLMs generate a block of tokens in parallel and iteratively
refine ("denoise") them, which can be very fast.

It is built specifically from the **Diffusion Gemma** llama.cpp pull request
([ggml-org/llama.cpp#24423](https://github.com/ggml-org/llama.cpp/pull/24423)),
following the [Unsloth Gemma Diffusion llama.cpp guide](https://unsloth.ai/docs/models/diffusiongemma#llama.cpp-guide).

> [!WARNING]
> This is an **experimental / custom** toolbox. It tracks an open llama.cpp PR
> and is built manually (not rebuilt automatically on upstream changes). The PR
> reference can be overridden at build time with `--build-arg PR=<number>` (set
> `PR=` empty to build plain `master` once the work is merged).

---

## 1. Create and Enter the Toolbox

> Ubuntu users: use `distrobox` instead of `toolbox`.

```sh
toolbox create llama-rocm-7.2.4-diffusion \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.4-diffusion \
  -- --device /dev/dri --device /dev/kfd --group-add video --group-add render --group-add sudo \
  --security-opt seccomp=unconfined

toolbox enter llama-rocm-7.2.4-diffusion
```

If you publish to your own registry, replace the image reference accordingly
(see [Building Locally](#4-building-locally)).

## 2. Download a Diffusion GGUF

Example: Unsloth's Gemma Diffusion GGUFs. Set `HF_TOKEN` for faster downloads.

```sh
pip install -U "huggingface_hub[cli]"
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF \
  --local-dir models/diffusiongemma-26B-A4B-it-GGUF \
  --include "*Q8_0*"   # Use "*Q4_K_M*" for a smaller (~16 GB) download
```

## 3. Run Inference

Conversational chat mode (`-cnv`), generating up to 2048 tokens:

```sh
llama-diffusion-cli \
  -m models/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q8_0.gguf \
  -ngl 999 -cnv -n 2048
```

Add `--diffusion-visual` to watch the live denoising of each token block:

```sh
llama-diffusion-cli \
  -m models/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q8_0.gguf \
  -ngl 999 -cnv -n 2048 --diffusion-visual
```

### Useful Diffusion Parameters

| Flag | Purpose |
| :--- | :--- |
| `--diffusion-steps` | Number of diffusion/refinement steps (default 256). |
| `--diffusion-block-length` | Block size for block-based scheduling (e.g. LLaDA). |
| `--diffusion-eps` | Epsilon for timestep-based scheduling (e.g. Dream). |
| `--diffusion-algorithm` | Token selection algorithm (see `llama-diffusion-cli --help`). |
| `--diffusion-visual` | Live visualization of the denoising process. |
| `-n` / `--n-predict` | Total number of tokens to generate. |
| `-ngl` | GPU layers to offload (use `999` to offload all on Strix Halo). |

Other diffusion architectures work too (build is the same CLI), for example:

```sh
# Dream
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" \
  -ub 512 --diffusion-eps 0.001 --diffusion-algorithm 3 --diffusion-steps 256 --diffusion-visual

# LLaDA
llama-diffusion-cli -m llada-8b.gguf -p "write code to train MNIST in pytorch" \
  -ub 512 --diffusion-block-length 32 --diffusion-steps 256 --diffusion-visual
```

## 4. Building Locally

```sh
cd toolboxes
podman build --no-cache -t llama-rocm-7.2.4-diffusion -f Dockerfile.rocm-7.2.4-diffusion .
```

Build options:

* `--build-arg PR=24423` — the llama.cpp PR to build (default). Set `PR=` empty
  to build plain `master`.
* `--build-arg REPO=...` / `--build-arg BRANCH=...` — alternate source.

See [building.md](building.md) for general build notes.

## 5. References

* [Unsloth — Gemma Diffusion (llama.cpp guide)](https://unsloth.ai/docs/models/diffusiongemma#llama.cpp-guide)
* [llama.cpp PR #24423 — DiffusionGemma](https://github.com/ggml-org/llama.cpp/pull/24423)
* [llama.cpp diffusion example README](https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion)
