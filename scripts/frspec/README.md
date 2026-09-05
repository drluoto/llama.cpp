# FR-Spec draft-vocab trim for the qwen4exp MTP sidecar

`trimma-sidovagn.py <in.gguf> <out.gguf> [K] [frequency-map.json]` keeps the K most frequent
rows of the sidecar's `output.weight` (Q8_0 rows gathered byte-for-byte) and adds a `d2t` (I64)
tensor mapping draft row -> vocabulary id. The loader in `src/models/qwen4exp.cpp` sizes
`output.weight` from `d2t` and scatters the compressed logits back to full-vocab shape in
`graph_mtp`'s head, so the target verifies over the full vocabulary and output is unchanged.

`tokenfrekvens2.json` is the map used for the published 65k sidecar (Qwen3.8-Flash-Next
tokenizer ids ranked by frequency: agent traces x5 + a small code/prose corpus). On Strix Halo
(Vulkan) the 65,536-row sidecar gave +7-9 % decode on real agent workloads at equal or better
acceptance; 32,768 rows lost acceptance on this workload. Measured with Claude Fable 5.1.
A Vitronia project.
