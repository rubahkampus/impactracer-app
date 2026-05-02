"""
Acid test: full E2E with remediated config and layer_compat.
Uses the superior doc chunks identified in diagnostic (endpoint-spec chunks
which contain exact function names and paths).
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
from impactracer.indexer.traceability import compute_and_store
from impactracer.shared.config import get_settings
from impactracer.shared.constants import layer_compat

settings = get_settings()
print(f"min_traceability_similarity = {settings.min_traceability_similarity}")
print(f"top_k_traceability          = {settings.top_k_traceability}")

# ── Build DB ───────────────────────────────────────────────────────────────
conn = sqlite3.connect(":memory:")
conn.execute("PRAGMA foreign_keys = ON")
init_schema(conn)

all_chunks = []
for md in ["srs.md", "sdd.md"]:
    all_chunks.extend(chunk_markdown(CITRAKARA_DOCS / md))
print(f"Chunks: {len(all_chunks)}")

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
total_edges = conn.execute("SELECT COUNT(*) FROM structural_edges").fetchone()[0]
print(f"Nodes: {total_nodes}  Edges: {total_edges}")

# ── Embed ──────────────────────────────────────────────────────────────────
print("Embedding...")
embedder = Embedder(settings.embedding_model, batch_size=32, max_length=512)

code_rows = conn.execute(
    "SELECT node_id, embed_text, file_classification FROM code_nodes "
    "WHERE embed_text IS NOT NULL AND LENGTH(embed_text) >= ?", (DEGENERATE_MIN,)
).fetchall()
print(f"Non-degenerate code nodes: {len(code_rows)}")

BATCH = 32
t0 = time.time()
parts = []
for i in range(0, len(code_rows), BATCH):
    parts.append(embedder.embed_batch([r[1] for r in code_rows[i:i+BATCH]]))
code_matrix = np.vstack(parts).astype(np.float32)

doc_matrix = embedder.embed_batch([c["text"] for c in all_chunks]).astype(np.float32)
print(f"Embedding done in {time.time()-t0:.1f}s")

code_vecs = {r[0]: code_matrix[i] for i, r in enumerate(code_rows)}
code_meta = {r[0]: {"file_classification": r[2]} for r in code_rows}
doc_vecs  = {c["chunk_id"]: doc_matrix[i] for i, c in enumerate(all_chunks)}
doc_meta  = {c["chunk_id"]: {"chunk_type": c["chunk_type"]} for c in all_chunks}

# ── Traceability ───────────────────────────────────────────────────────────
print("Computing traceability...")
n_pairs = compute_and_store(
    code_vecs, doc_vecs, code_meta, doc_meta,
    top_k=settings.top_k_traceability,
    min_similarity=settings.min_traceability_similarity,
    conn=conn,
)
print(f"Pairs stored: {n_pairs}")

# ── Global metrics ─────────────────────────────────────────────────────────
scores = [r[0] for r in conn.execute(
    "SELECT weighted_similarity_score FROM doc_code_candidates"
).fetchall()]
arr = np.array(scores) if scores else np.array([0.0])

print("\n" + "="*70)
print("GLOBAL METRICS")
print("="*70)
print(f"  Total Doc Chunks:                              {len(all_chunks)}")
print(f"  Total Code Nodes (all):                        {total_nodes}")
print(f"  Total Code Nodes (non-degenerate embedded):    {len(code_rows)}")
print(f"  Total Edges:                                   {total_edges}")
print(f"  Traceability Candidates (doc_code_candidates): {n_pairs}")
print(f"  Score min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}"
      f"  median={np.median(arr):.4f}  std={arr.std():.4f}")
print("  Distribution:")
for lo, hi in [(0.40,0.45),(0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.01)]:
    c = int(((arr>=lo)&(arr<hi)).sum())
    print(f"    [{lo:.2f},{hi:.2f}): {c}")

# ── Deep-dive on 3 benchmark chunks ───────────────────────────────────────
# Use the superior endpoint-spec chunks found during diagnostic
BENCHMARK = [
    # (theme, chunk_id_substring, description)
    ("Autentikasi/Login",
     "v_1_perancangan_antarmuka_layanan_endpoint_autentikasi",
     "SDD endpoint spec — explicitly names loginUser::auth.service, POST /api/auth/login"),
    ("Wallet/Saldo",
     "v_17_perancangan_antarmuka_layanan_endpoint_wallet",
     "SDD endpoint spec — names getWalletSummary::wallet.service, GET /api/wallet/balance"),
    ("Sengketa/Dispute",
     "iii_perancangan_komponen_modul_sengketa",
     "SDD dispute module — fallback to keyword search"),
    # Sprint 6.5 NEW — DB Design chunk should resolve to wallet.model.ts
    ("DB Design — Entitas Wallet",
     "iv_2_perancangan_basis_data_entitas_wallet",
     "SDD DB design chunk — TYPE_DEFINITION×Design=1.0 should surface wallet.model.ts"),
]

# Also test the original M01/M04 chunks to measure improvement
ORIGINAL = [
    ("Auth (original M01)", "iii_1_perancangan_komponen_modul_autentikasi_m01"),
    ("Wallet (original M04)", "iii_3_perancangan_komponen_modul_wallet_m04"),
    ("Dispute (original V4)", "v_4_deskripsi_proses_bisnis_pengerjaan"),
]

print("\n" + "="*70)
print("TRACEABILITY DEEP-DIVE — Preferred endpoint-spec chunks")
print("="*70)

def find_chunk(substring):
    for c in all_chunks:
        if substring in c["chunk_id"]:
            return c
    # keyword fallback
    kws = substring.replace("_", " ").split()
    best, best_hits = None, -1
    for c in all_chunks:
        hits = sum(1 for kw in kws if kw in (c["section_title"]+" "+c["text"]).lower())
        if hits > best_hits:
            best_hits, best_hits_chunk = hits, best_hits
            best = c
            best_hits = hits
    return best

def show_top3(cid, title, chunk_type, note=""):
    top3 = conn.execute(
        """SELECT dcc.code_id, dcc.weighted_similarity_score,
                  cn.node_type, cn.name, cn.file_path, cn.file_classification
           FROM doc_code_candidates dcc
           JOIN code_nodes cn ON cn.node_id = dcc.code_id
           WHERE dcc.doc_id = ?
           ORDER BY dcc.weighted_similarity_score DESC LIMIT 3""",
        (cid,)
    ).fetchall()
    print(f"\n  Chunk: {cid}")
    print(f"  Title: {title}")
    print(f"  Type:  {chunk_type}  {note}")
    if not top3:
        # Show why — what are the raw scores?
        print("  [NO CANDIDATES — checking raw scores...]")
        # compute raw cosine manually
        if cid in doc_vecs:
            dv = doc_vecs[cid]
            dv_n = dv / (np.linalg.norm(dv) or 1.0)
            sims = []
            for r in code_rows:
                cv = code_vecs[r[0]]
                cv_n = cv / (np.linalg.norm(cv) or 1.0)
                raw = float(np.dot(cv_n, dv_n))
                w = raw * layer_compat(r[2], chunk_type)
                sims.append((w, raw, r[0], r[2]))
            sims.sort(reverse=True)
            print(f"  Top-3 weighted (unfiltered): threshold was {settings.min_traceability_similarity}")
            for w, raw, nid, cls in sims[:3]:
                print(f"    weighted={w:.4f}  raw={raw:.4f}  class={cls}  {nid}")
    else:
        for rank, (nid, score, ntype, name, fpath, fclass) in enumerate(top3, 1):
            print(f"  Top {rank}: [{score:.4f}]  {nid}")
            print(f"           type={ntype}  name={name}  class={fclass}")

print("\n--- PREFERRED CHUNKS (endpoint-spec) ---")
for theme, substr, note in BENCHMARK:
    c = find_chunk(substr)
    if c:
        print(f"\n=== {theme} ===")
        show_top3(c["chunk_id"], c["section_title"], c["chunk_type"], note)
    else:
        print(f"\n=== {theme} === [chunk not found]")

print("\n--- ORIGINAL CHUNKS (module description) ---")
for theme, substr in ORIGINAL:
    c = find_chunk(substr)
    if c:
        print(f"\n=== {theme} ===")
        show_top3(c["chunk_id"], c["section_title"], c["chunk_type"], "(original diagnostic chunk)")
    else:
        print(f"\n=== {theme} === [chunk not found]")

# ── Sprint 6.5 Pass/Fail verdict ────────────────────────────────────────────
print("\n" + "="*70)
print("SPRINT 6.5 ACID TEST — PASS/FAIL VERDICTS")
print("="*70)

EXPECTED = {
    "v_1_perancangan_antarmuka_layanan_endpoint_autentikasi":
        ("auth", ["auth.service", "auth", "login"]),
    "v_17_perancangan_antarmuka_layanan_endpoint_wallet":
        ("wallet", ["wallet.service", "wallet", "balance"]),
    "iii_perancangan_komponen_modul_sengketa":
        ("dispute/resolution", ["dispute", "resolution", "sengketa"]),
    "iv_2_perancangan_basis_data_entitas_wallet":
        ("wallet model", ["wallet.model", "wallet"]),
}

all_pass = True
for substr, (label, keywords) in EXPECTED.items():
    c = find_chunk(substr)
    if c is None:
        print(f"  FAIL  [{label}] chunk not found")
        all_pass = False
        continue
    top3 = conn.execute(
        """SELECT dcc.code_id, dcc.weighted_similarity_score,
                  cn.node_type, cn.name, cn.file_path, cn.file_classification
           FROM doc_code_candidates dcc
           JOIN code_nodes cn ON cn.node_id = dcc.code_id
           WHERE dcc.doc_id = ?
           ORDER BY dcc.weighted_similarity_score DESC LIMIT 3""",
        (c["chunk_id"],)
    ).fetchall()
    if not top3:
        print(f"  FAIL  [{label}] no candidates")
        all_pass = False
        continue
    top_node = top3[0]
    nid_lower = top_node[0].lower()
    fpath_lower = top_node[4].lower() if top_node[4] else ""
    hit = any(kw in nid_lower or kw in fpath_lower for kw in keywords)
    status = "PASS" if hit else "WARN"
    if not hit:
        all_pass = False
    print(f"  {status}  [{label}] top-1={top_node[1]:.4f}  {top_node[0]}")
    print(f"           file_path={top_node[4]}  class={top_node[5]}")

print()
print("  OVERALL:", "ALL PASS ✓" if all_pass else "SOME FAIL — review above")

# ── layer_compat check ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("LAYER_COMPAT DISTRIBUTION CHECK")
print("="*70)
rows = conn.execute(
    """SELECT cn.file_classification, AVG(dcc.weighted_similarity_score),
              COUNT(*), MIN(dcc.weighted_similarity_score), MAX(dcc.weighted_similarity_score)
       FROM doc_code_candidates dcc
       JOIN code_nodes cn ON cn.node_id = dcc.code_id
       GROUP BY cn.file_classification
       ORDER BY AVG(dcc.weighted_similarity_score) DESC"""
).fetchall()
for fclass, avg, cnt, mn, mx in rows:
    print(f"  {str(fclass):20s}  avg={avg:.4f}  min={mn:.4f}  max={mx:.4f}  count={cnt}")

print("\n=== ACID TEST COMPLETE ===")
conn.close()
