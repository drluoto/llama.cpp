#!/usr/bin/env python3
"""Spektrum av arbetslaster mot en lopande server (temp 0, cache_prompt=false):
kort kod @0, ny kod @8k, omskrivning @8k, prosa @8k, ny kod @32k, omskrivning @32k.
Skriver prefill t/s, decode t/s, tok/steg, acceptans, ms/steg per last."""
import json, sys, urllib.request, os
PORT = os.environ.get("PORT", "8098")
BAS = ("Minnesbandbredd begransar avkodningen pa integrerade GPU:er eftersom varje token kraver att "
       "alla aktiva vikter lases fran DRAM, medan berakningen per token ar liten. ")
def post(path, d):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
def ntok(s): return len(post("/tokenize", {"content": s})["tokens"])
def fyll(D):
    if D <= 0: return ""
    per = ntok(BAS); f = BAS * (D // per)
    while ntok(f) < D - 40: f += BAS
    while ntok(f) > D - 20: f = f[:-len(BAS)]
    return f
def pyfil(D):
    """syntetisk Python-fil pa ~D tokens"""
    delar = []; i = 0; txt = ""
    while ntok(txt) < D:
        delar.append(f'''def func_{i}(items, limit={i % 7 + 1}):
    """Return the first `limit` items whose score exceeds {i * 3 % 11}."""
    out = []
    for item in items:
        if item.score > {i * 3 % 11} and len(out) < limit:
            out.append(item.name.strip().lower())
    return out


''')
        i += 1
        if i % 10 == 0: txt = "".join(delar)
    return "".join(delar)
KOD = "\n\nSkriv en komplett Python-klass for en LRU-cache med get/put, docstrings och typannoteringar."
PROSA = "\n\nSkriv en essa pa 600 ord om varfor lokala sprakmodeller spelar roll for smaforetag. Vanlig prosa, inga listor."
def omskriv(fil): return "Here is a Python file:\n\n```python\n" + fil + "```\n\nReturn the complete file unchanged except rename every function `func_` to `fn_`. Output only the code, no commentary."
laster = [
    ("kort kod @0",      lambda: fyll(0) + KOD, 256),
    ("ny kod @8k",       lambda: fyll(8192) + KOD, 512),
    ("prosa @8k",        lambda: fyll(8192) + PROSA, 512),
    ("omskrivning @8k",  lambda: omskriv(pyfil(7800)), 1024),
    ("ny kod @32k",      lambda: fyll(32768) + KOD, 512),
    ("omskrivning @32k", lambda: fyll(24000) + "\n\n" + omskriv(pyfil(7800)), 1024),
]
val = sys.argv[1:]
for namn, bygg, mx in laster:
    if val and not any(v in namn for v in val): continue
    p = bygg()
    d = {"model": "q", "messages": [{"role": "user", "content": p}], "max_tokens": mx, "temperature": 0,
         "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    t = post("/v1/chat/completions", d)["timings"]
    n = t["predicted_n"]; ms = t["predicted_ms"]; dn = t.get("draft_n", 0); da = t.get("draft_n_accepted", 0); steg = n - da
    print(f"{namn:17s} prompt_n={t['prompt_n']:6d} prefill={t['prompt_n']/t['prompt_ms']*1000:6.1f} t/s | decode={n/ms*1000:5.1f} t/s n={n}"
          + (f" tok/steg={n/max(steg,1):.2f} accept={da/dn:.2f} ms/steg={ms/max(steg,1):.1f}" if dn else f" ms/token={ms/n:.1f}"), flush=True)
