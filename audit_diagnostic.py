"""
Deep diagnostic sweep for traceability failure.
Tests four hypotheses without modifying production code.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

CITRAKARA_SRC  = Path(r"C:\Users\Haidar\Documents\thesis\citrakara\src")
CITRAKARA_DOCS = Path(r"C:\Users\Haidar\Documents\thesis\citrakara\docs")
DEGENERATE_MIN = 50

from impactracer.persistence.sqlite_client import init_schema
from impactracer.indexer.doc_indexer import chunk_markdown
from impactracer.indexer.code_indexer import extract_nodes, extract_edges
from impactracer.indexer.embedder import Embedder
from impactracer.shared.config import get_settings
from impactracer.shared.constants import layer_compat

settings = get_settings()

# ── Build DB ───────────────────────────────────────────────────────────────
print("Building DB + indexing (reusing cached model)...")
conn = sqlite3.connect(":memory:")
conn.execute("PRAGMA foreign_keys = ON")
init_schema(conn)

all_chunks = []
for md in ["srs.md", "sdd.md"]:
    all_chunks.extend(chunk_markdown(CITRAKARA_DOCS / md))

conn.execute("CREATE TABLE IF NOT EXISTS _dc (chunk_id TEXT PRIMARY KEY, section_title TEXT, chunk_type TEXT, text TEXT)")
conn.executemany("INSERT OR IGNORE INTO _dc VALUES(?,?,?,?)",
                 [(c["chunk_id"],c["section_title"],c["chunk_type"],c["text"]) for c in all_chunks])
conn.commit()

ts_files = sorted(CITRAKARA_SRC.rglob("*.ts")) + sorted(CITRAKARA_SRC.rglob("*.tsx"))
for f in ts_files:
    extract_nodes(f, f.read_bytes(), conn)
known = {r[0] for r in conn.execute("SELECT node_id FROM code_nodes")}
for f in ts_files:
    extract_edges(f, f.read_bytes(), known, conn)

total_nodes = conn.execute("SELECT COUNT(*) FROM code_nodes").fetchone()[0]
print(f"  nodes={total_nodes}  chunks={len(all_chunks)}")

# ── Embed everything ───────────────────────────────────────────────────────
print("Embedding (this is the slow step)...")
embedder = Embedder(settings.embedding_model, batch_size=32, max_length=512)

code_rows = conn.execute(
    "SELECT node_id, embed_text, file_classification, node_type, name FROM code_nodes "
    "WHERE embed_text IS NOT NULL AND LENGTH(embed_text) >= ?", (DEGENERATE_MIN,)
).fetchall()

code_texts = [r[1] for r in code_rows]
chunks_text = [c["text"] for c in all_chunks]

t0 = time.time()
BATCH = 32
code_vecs_list = []
for i in range(0, len(code_texts), BATCH):
    code_vecs_list.append(embedder.embed_batch(code_texts[i:i+BATCH]))
code_matrix = np.vstack(code_vecs_list).astype(np.float32)

doc_matrix = embedder.embed_batch(chunks_text).astype(np.float32)
print(f"  Embedding done in {time.time()-t0:.1f}s  code={code_matrix.shape}  doc={doc_matrix.shape}")

# ── HYPOTHESIS 1: Raw cosine distribution (no layer_compat, no threshold) ──
print("\n=== HYPOTHESIS 1: Raw cosine distribution ===")
# L2-normalize
def l2norm(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return m / n

cn = l2norm(code_matrix)
dn = l2norm(doc_matrix)
cos = cn @ dn.T   # (N_code, N_doc)

flat = cos.flatten()
print(f"  Cosine scores (all {len(flat)} pairs):")
print(f"    min={flat.min():.4f}  max={flat.max():.4f}  mean={flat.mean():.4f}"
      f"  median={np.median(flat):.4f}  std={flat.std():.4f}")

buckets = [(-1.0,-0.2),(-0.2,0.0),(0.0,0.2),(0.2,0.4),(0.4,0.5),(0.5,0.6),
           (0.6,0.65),(0.65,0.7),(0.7,0.8),(0.8,1.01)]
print("  Distribution:")
for lo,hi in buckets:
    c = int(((flat>=lo)&(flat<hi)).sum())
    pct = 100*c/len(flat)
    print(f"    [{lo:5.2f},{hi:5.2f}): {c:7d}  ({pct:.2f}%)")

# ── HYPOTHESIS 2: Per-chunk max cosine for our 3 benchmark chunks ───────────
print("\n=== HYPOTHESIS 2: Raw cosine for 3 benchmark chunks ===")
THEMES = {
    "Auth":    ["autentikasi","login","auth","masuk","register"],
    "Wallet":  ["wallet","saldo","balance","pembayaran","topup","top up","dana"],
    "Dispute": ["sengketa","dispute","eskro","escrow","komplain","refund"],
}
for theme, kws in THEMES.items():
    best_i, best_hits = None, -1
    for i, c in enumerate(all_chunks):
        combined = (c["section_title"]+" "+c["text"]).lower()
        hits = sum(1 for kw in kws if kw in combined)
        if hits > best_hits:
            best_hits, best_i = hits, i
    if best_i is None:
        print(f"  [{theme}] no chunk found"); continue
    c = all_chunks[best_i]
    row = cos[:, best_i]          # cosine of every code node against this doc chunk
    top5_idx = np.argsort(row)[::-1][:5]
    print(f"\n  [{theme}] chunk={c['chunk_id']} (hits={best_hits})")
    print(f"    max_cosine={row.max():.4f}  mean={row.mean():.4f}  std={row.std():.4f}")
    print(f"    Top-5 raw cosine code nodes:")
    for rank, idx in enumerate(top5_idx, 1):
        r = code_rows[idx]
        print(f"      {rank}. cos={row[idx]:.4f}  {r[0]}  type={r[3]}  name={r[4]}")

# ── HYPOTHESIS 3: layer_compat suppression analysis ─────────────────────────
print("\n=== HYPOTHESIS 3: layer_compat suppression on benchmark chunks ===")
for theme, kws in THEMES.items():
    best_i, best_hits = None, -1
    for i, c in enumerate(all_chunks):
        combined = (c["section_title"]+" "+c["text"]).lower()
        hits = sum(1 for kw in kws if kw in combined)
        if hits > best_hits:
            best_hits, best_i = hits, i
    if best_i is None: continue
    chunk = all_chunks[best_i]
    ctype = chunk["chunk_type"]
    row = cos[:, best_i]
    top_idx = np.argsort(row)[::-1][:10]
    print(f"\n  [{theme}] chunk_type={ctype}")
    for idx in top_idx:
        r = code_rows[idx]
        raw = float(row[idx])
        compat = layer_compat(r[2], ctype)
        weighted = raw * compat
        print(f"    raw={raw:.4f}  compat={compat:.2f}  weighted={weighted:.4f}  "
              f"class={r[2]}  {r[0]}")

# ── HYPOTHESIS 4: Anisotropy — mean cosine between random code pairs ─────────
print("\n=== HYPOTHESIS 4: Embedding anisotropy check ===")
rng = np.random.default_rng(42)
idx_a = rng.integers(0, len(cn), 500)
idx_b = rng.integers(0, len(cn), 500)
inter_code = (cn[idx_a] * cn[idx_b]).sum(axis=1)
print(f"  Mean cosine between 500 random code-code pairs: {inter_code.mean():.4f}  std={inter_code.std():.4f}")

idx_a2 = rng.integers(0, len(dn), 50)
idx_b2 = rng.integers(0, len(dn), 50)
inter_doc = (dn[idx_a2] * dn[idx_b2]).sum(axis=1)
print(f"  Mean cosine between 50 random doc-doc pairs:    {inter_doc.mean():.4f}  std={inter_doc.std():.4f}")

cross_sample = cos[idx_a[:50], :][:, :50].flatten()
print(f"  Mean cosine cross (code-doc sample):            {cross_sample.mean():.4f}  std={cross_sample.std():.4f}")

# ── HYPOTHESIS 5: embed_text quality spot-check ──────────────────────────────
print("\n=== HYPOTHESIS 5: embed_text spot-check for auth/wallet nodes ===")
auth_nodes = conn.execute(
    "SELECT node_id, node_type, name, embed_text FROM code_nodes "
    "WHERE (node_id LIKE '%auth%' OR node_id LIKE '%login%' OR node_id LIKE '%sign%') "
    "AND embed_text IS NOT NULL AND LENGTH(embed_text) >= ? LIMIT 5", (DEGENERATE_MIN,)
).fetchall()
wallet_nodes = conn.execute(
    "SELECT node_id, node_type, name, embed_text FROM code_nodes "
    "WHERE (node_id LIKE '%wallet%' OR node_id LIKE '%balance%' OR node_id LIKE '%payment%') "
    "AND embed_text IS NOT NULL AND LENGTH(embed_text) >= ? LIMIT 5", (DEGENERATE_MIN,)
).fetchall()

print(f"  Auth-related nodes ({len(auth_nodes)}):")
for nid, ntype, name, et in auth_nodes:
    print(f"    {nid}  type={ntype}  name={name}")
    print(f"      embed_text[:200]: {et[:200].replace(chr(10),' ')}")

print(f"\n  Wallet-related nodes ({len(wallet_nodes)}):")
for nid, ntype, name, et in wallet_nodes:
    print(f"    {nid}  type={ntype}  name={name}")
    print(f"      embed_text[:200]: {et[:200].replace(chr(10),' ')}")

# ── HYPOTHESIS 6: What threshold would give ~1000-5000 pairs? ───────────────
print("\n=== HYPOTHESIS 6: Threshold calibration ===")
from impactracer.shared.constants import LAYER_COMPAT
compat_arr = np.empty((len(code_rows), len(all_chunks)), dtype=np.float32)
for i, r in enumerate(code_rows):
    for j, c in enumerate(all_chunks):
        compat_arr[i,j] = layer_compat(r[2], c["chunk_type"])
weighted_all = cos * compat_arr

wflat = weighted_all.flatten()
for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    cnt = int((wflat >= thresh).sum())
    print(f"  min_similarity={thresh:.2f}: {cnt} pairs ({100*cnt/len(wflat):.3f}%)")

print("\n=== DIAGNOSTIC COMPLETE ===")
conn.close()
