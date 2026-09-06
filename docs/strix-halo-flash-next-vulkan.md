# Qwen3.8-Flash-Next on Strix Halo with Vulkan

This branch is the exact llama.cpp build I run on a Bosgame M5 (Ryzen AI Max+ 395, Radeon 8060S, 128 GB) as the model server for my agents. It is a fork of apepojken's `qwen4exp-spec-mtp` lineage with the changes below on top. Vulkan only, RADV, Mesa 26.0.3. No ROCm.

Measured on 10 real agent conversations replayed against the server (medians), plus a single-stream ladder on a code prompt:

| what | number |
|---|---:|
| decode, warm cache | 33.1 tok/s |
| decode, cold | 32.5 tok/s |
| cold time to first token, ~18k-token prompt | 44 s |
| prefill, 8k prompt | 517 tok/s |
| prefill, 32k prompt | 391 tok/s |
| decode at 8k, single stream, code | 41 tok/s |

Without speculation the same trunk decodes at about 27 tok/s, which is where the memory bandwidth of this box puts it. Greedy output is bit-identical between runs.

## What is in the branch

Relative to plain llama.cpp master, in order of effect:

1. **Row-id hoisting for 512 experts.** Upstream turns the hoisted row-id path off above 256 experts, so every expert matmul workgroup rescanned the routing tensor. Raising the limit gave +19 % prefill. Sent upstream as ggml-org/llama.cpp#28501.
2. **FR-Spec trimmed MTP draft head.** The draft head's vocabulary is cut to the 65 536 most frequent tokens with a `d2t` map back to real ids. The trunk still verifies over the full vocabulary, so the text is unchanged. Loader support is in `src/models/qwen4exp.cpp`, the trim script in `scripts/frspec/`. Draft head file: https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF
3. **Always draft the full three tokens** (`--spec-draft-p-min 0.0`). Extra draft tokens cost about 4 ms each on this box, so early-stopping the drafter loses more than it saves. +13 % decode.
4. **Deterministic speculation.** KV cells are zeroed when freed (from nathanw1014) and the ported GDN state-cache fusion is disabled with `GGML_VK_DISABLE_GDN_CACHE_FUSION=1`, which fixed non-deterministic output under `-np 3` with MTP.
5. Smaller things: LDS pad 2 for the coopmat tiles on RADV >= 25.3 (from nathanw1014), a tiled transpose kernel for the delta-net conv concat, and `-ub 2048`.

Things I tried that did not help on this hardware, so you do not have to: fusing the small glue kernels (dispatch count is not the bottleneck here), a grouped expert kernel that shares weight reads between the speculative tokens (the Infinity Cache already does that), and draft lengths above three.

## Model files

Trunk: a requant of Unsloth's UD-IQ4_XS with the dense weights at Q5_K and the routers at Q8_0, experts left as they are. Recipe, from the UD-IQ4_XS shards:

```sh
LLAMA_QUANT_ALLOW_ROUTER=1 llama-quantize --allow-requantize \
  --tensor-type-file scripts/strix-halo/tensor-types-q5k.txt --keep-split \
  Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf trunk-q5k.gguf Q5_K_M 12
```

The stock UD-IQ4_XS works too and gives the prefill gain; the decode numbers above were measured with the requant.

Draft head: `mtp-Qwen3.8-Flash-Next-Q8_0-frspec-65k.gguf` from the HF repo above. Needs this branch; stock llama.cpp will not load it.

## Build and run

```sh
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j --target llama-server

GGML_VK_DISABLE_GDN_CACHE_FUSION=1 build/bin/llama-server \
  -m trunk-q5k-00001-of-00003.gguf \
  -md mtp-Qwen3.8-Flash-Next-Q8_0-frspec-65k.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0 \
  -fa 1 -ub 2048 -b 2048 -c 262144 -np 3 --ctx-checkpoints 8 \
  -ctk f16 -ctv f16 -lm dio --jinja
```

`-np 3 --ctx-checkpoints 8` matters if more than one session shares the server: each session keeps its own warm cache instead of evicting the others. Checkpoints are required for rollback on this architecture.

## Credits

apepojken for the MTP lineage, nathanw1014 for the KV zeroing and the RADV tuning, avifenesh for the original FR-Spec work (#25187). The profiling and changes here were done together with Claude Fable 5.1 in Claude Code; I reviewed and measured everything on the box. A Vitronia project.
