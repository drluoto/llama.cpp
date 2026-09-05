#!/usr/bin/env python3
"""FR-Spec-trimmad MTP-sidovagn: output.weight reduceras till K frekvensrankade rader + d2t (I64, mal-id
per utkastrad). Allt annat kopieras byte for byte. Rankning: arbetslastens tokenfrekvens forst, sedan
stigande id (BPE-mergeordning ~ frekvens).  anvandning: trimma-sidovagn.py <in> <ut> [K]"""
import sys, json, numpy as np
sys.path.insert(0, "/data/llama.cpp-merge/gguf-py")
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType
src, dst = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
reader = GGUFReader(src)
n_vocab = next(int(t.shape[1]) for t in reader.tensors if t.name == "output.weight")
KARTA = sys.argv[4] if len(sys.argv) > 4 else "/data/bank/tokenfrekvens.json"
rank = json.load(open(KARTA))["rank"]
keep, seen = [], set()
for i in rank:
    if i < n_vocab and i not in seen: keep.append(i); seen.add(i)
for i in range(n_vocab):
    if len(keep) >= K: break
    if i not in seen: keep.append(i); seen.add(i)
keep = np.array(sorted(keep[:K]), dtype=np.int64)
print(f"n_vocab {n_vocab} -> K {len(keep)} (fran arbetslasten: {len(set(rank) & set(keep.tolist()))})")
arch = reader.fields["general.architecture"].contents()
w = GGUFWriter(dst, arch=arch, endianess=reader.endianess)
for f in reader.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture": continue
    vt = f.types[0]; st = f.types[-1] if vt == GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
w.add_string("general.frspec.note", f"output.weight trimmed to {len(keep)} frequency-ranked rows; d2t maps draft row -> target id")
plan = []
for t in reader.tensors:
    if t.name == "output.weight":
        data = np.ascontiguousarray(t.data[keep]); print(f"output.weight {t.data.shape} -> {data.shape}")
    else:
        data = t.data
    w.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type); plan.append(data)
d2t = keep.copy(); w.add_tensor_info("d2t", d2t.shape, d2t.dtype, d2t.nbytes, gguf.GGMLQuantizationType.I64); plan.append(d2t)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for data in plan: w.write_tensor_data(data, tensor_endianess=reader.endianess)
w.close()
r2 = GGUFReader(dst); print("kontroll:", [(t.name, t.tensor_type.name, list(map(int, t.shape))) for t in r2.tensors if t.name in ("output.weight", "d2t", "token_embd.weight")])
