import os, sys, re, json, time, random, hashlib, unicodedata
from pathlib import Path
from collections import defaultdict, Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import networkx as nx

dotenv.load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def get_secret(name, default=None):
    return os.environ.get(name, default)

NEO4J_URI = get_secret("NEO4J_URI", "")
NEO4J_USER = get_secret("NEO4J_USER", get_secret("NEO4J_USERNAME", "neo4j"))
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "")
NEO4J_DATABASE = get_secret("NEO4J_DATABASE", "neo4j")

GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
if GROQ_MODEL == "qwen/qwen3.6-27b":
    GROQ_MODEL = "openai/gpt-oss-120b"

JUDGE_PROVIDER = "groq"
JUDGE_MODEL = "openai/gpt-oss-120b"

DATA_PATH = "outputs/tech-news.csv"
LAB_MAX_ARTICLES = 1500
LAB_MAX_CHUNKS = 3000
EXTRACTION_MAX_CHUNKS = 300
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

SUPER_NODE_DEGREE = 100
SUPER_NODE_LIMIT = 50
GLOBAL_EDGE_CAP = 250
MAX_GRAPH_CONTEXT_CHARS = 14000

print(f"Configs loaded:", flush=True)
print(f"Neo4j: {NEO4J_URI} (User: {NEO4J_USER})", flush=True)
print(f"Groq Model: {GROQ_MODEL}", flush=True)
print(f"Judge: {JUDGE_PROVIDER} / {JUDGE_MODEL}", flush=True)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def parse_json_object(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(text[a:b+1])

def groq_chat(messages, model=None, json_mode=False, max_retries=5):
    if groq_client is None:
        raise RuntimeError("Missing GROQ_API_KEY.")
    models_to_try = [model or GROQ_MODEL, "openai/gpt-oss-20b", "groq/compound-mini"]
    last_err = None
    for attempt in range(max_retries):
        m = models_to_try[attempt % len(models_to_try)]
        try:
            kwargs = {
                "model": m,
                "messages": messages,
                "temperature": 0.0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = groq_client.chat.completions.create(**kwargs)
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            return resp.choices[0].message.content, usage
        except Exception as e:
            last_err = e
            time.sleep(min(10, 0.5 * (2**attempt) + random.uniform(0.1, 0.4)))
    raise RuntimeError(f"groq_chat failed after {max_retries} attempts: {last_err}")

def groq_json(system, user, model=None):
    text, usage = groq_chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        model=model,
        json_mode=True
    )
    return parse_json_object(text), usage

# ==========================================
# 2. NEO4J CONNECTION & SCHEMA
# ==========================================
GLOBAL_DRIVER = None

def get_driver():
    global GLOBAL_DRIVER
    if GLOBAL_DRIVER is None:
        GLOBAL_DRIVER = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            keep_alive=True
        )
    return GLOBAL_DRIVER

def run_cypher(query, **params):
    for attempt in range(4):
        try:
            driver = get_driver()
            with driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(query, **params)
                return [record.data() for record in result]
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
            # Recreate driver on connection failure
            global GLOBAL_DRIVER
            try:
                if GLOBAL_DRIVER:
                    GLOBAL_DRIVER.close()
            except:
                pass
            GLOBAL_DRIVER = None

def setup_graph_schema():
    print("Setting up Neo4j schema & constraints...", flush=True)
    run_cypher("MATCH (n) DETACH DELETE n")
    try:
        run_cypher("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
    except Exception:
        pass
    try:
        run_cypher("CREATE INDEX entity_name_norm IF NOT EXISTS FOR (n:Entity) ON (n.name_norm)")
    except Exception:
        pass
    print("✅ Neo4j schema ready & graph cleaned.", flush=True)

# ==========================================
# 3. TEXT UTILS & DEDUP & CHUNKING
# ==========================================
def norm_space(x):
    if not isinstance(x, str):
        return ""
    x = unicodedata.normalize("NFKC", x)
    return re.sub(r"\s+", " ", x).strip()

def sha1(x):
    return hashlib.sha1(str(x).encode("utf-8", errors="ignore")).hexdigest()

def chunk_text(text, chunk_words=CHUNK_WORDS, overlap_words=CHUNK_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for i in range(0, len(words), step):
        w = words[i:i + chunk_words]
        chunks.append(" ".join(w))
        if i + chunk_words >= len(words):
            break
    return chunks

def build_chunks(df_articles, max_chunks=LAB_MAX_CHUNKS):
    records = []
    for art_idx, row in df_articles.iterrows():
        title = norm_space(row.get("title", ""))
        desc = norm_space(row.get("description", ""))
        full_text = f"{title}. {desc}".strip()
        pub_date = str(row.get("published_at", "")).strip()
        company = norm_space(row.get("companyName", ""))
        url = str(row.get("url", "")).strip()

        c_list = chunk_text(full_text)
        for c_idx, c_txt in enumerate(c_list):
            chunk_id = f"art_{art_idx:04d}::c{c_idx:04d}"
            records.append({
                "chunk_id": chunk_id,
                "article_id": art_idx,
                "title": title,
                "published_at": pub_date,
                "companyName": company,
                "url": url,
                "text": c_txt,
                "word_count": len(c_txt.split())
            })
            if len(records) >= max_chunks:
                break
        if len(records) >= max_chunks:
            break
    return pd.DataFrame(records)

# ==========================================
# 4. COREFERENCE RESOLUTION (MODULE 1)
# ==========================================
COREF_SYSTEM = """
You are a conservative coreference resolution system for tech business news.
Given a list of text chunks, resolve ambiguous third-person pronouns (it, they, he, she, the company, the startup) to their explicit named entity antecedent ONLY when the antecedent is clearly present in the same chunk.
Rules:
1. Conservative rule: Do NOT guess or hallucinate. If ambiguous or missing, keep original text.
2. Return a strict JSON object with a list of resolved chunks:
{"results": [{"chunk_id": "...", "resolved_text": "...", "unresolved_mentions": ["..."]}]}
""".strip()

def process_coref_batch(batch_items):
    items_payload = [{"chunk_id": item["chunk_id"], "text": item["text"]} for item in batch_items]
    user_prompt = f"CHUNKS TO RESOLVE:\n{json.dumps(items_payload, ensure_ascii=False)}"
    try:
        res_json, _ = groq_json(COREF_SYSTEM, user_prompt)
        results = res_json.get("results", [])
        return {r["chunk_id"]: (r.get("resolved_text", ""), r.get("unresolved_mentions", [])) for r in results}
    except Exception:
        return {}

def run_coreference_resolution(chunks_df, max_chunks=EXTRACTION_MAX_CHUNKS, batch_size=5, max_workers=8):
    print(f"\n--- Running Conservative Coreference Resolution on {min(len(chunks_df), max_chunks)} chunks (Parallel Workers: {max_workers}) ---", flush=True)
    subset = chunks_df.head(max_chunks).copy()
    items = subset.to_dict(orient="records")
    
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    results_map = {}
    unresolved_all = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_coref_batch, b): b for b in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Coreference Batches"):
            res = future.result()
            for cid, (txt, unres) in res.items():
                results_map[cid] = txt
                unresolved_all.extend(unres)

    resolved_list = [results_map.get(r["chunk_id"], r["text"]) for r in items]
    subset["resolved_text"] = resolved_list
    print(f"✅ Coreference Resolution complete. Logged {len(unresolved_all)} unresolved mentions.", flush=True)
    return subset, unresolved_all

# ==========================================
# 5. TRIPLE EXTRACTION (MODULE 2)
# ==========================================
ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED", "DEVELOPED", "INVESTED_IN", "FOUNDED",
    "WORKED_AT", "PARTNERED_WITH", "USES", "LEADS"
}

EXTRACTION_SYSTEM = """
You are an expert knowledge graph extraction engine for tech business news.
Extract factual relationships matching the exact schema:
Allowed Node Types: Company, Person, Technology
Allowed Relations:
- ACQUIRED (Company -> Company)
- DEVELOPED (Company/Person -> Technology)
- INVESTED_IN (Company/Person -> Company)
- FOUNDED (Person/Company -> Company)
- WORKED_AT (Person -> Company)
- PARTNERED_WITH (Company -> Company)
- USES (Company -> Technology)
- LEADS (Person -> Company)

For every relationship:
- source_raw: exact name of source entity
- source_type: Company | Person | Technology
- relation: exact uppercase relation from allowed list
- target_raw: exact name of target entity
- target_type: Company | Person | Technology
- evidence: direct quote from the chunk as evidence
- confidence: float 0.0 - 1.0

Return strict JSON:
{"triples": [{"source_raw": "...", "source_type": "...", "relation": "...", "target_raw": "...", "target_type": "...", "evidence": "...", "confidence": 0.95}]}
""".strip()

def extract_single_chunk(row_dict):
    cid = row_dict["chunk_id"]
    pub_date = str(row_dict.get("published_at", ""))[:10] or "2023-01-01"
    prompt = f"CHUNK ID: {cid}\nDATE: {pub_date}\nTEXT:\n{row_dict['resolved_text']}"
    triples_out = []
    try:
        res, _ = groq_json(EXTRACTION_SYSTEM, prompt)
        for t in res.get("triples", []):
            rel = str(t.get("relation", "")).strip().upper()
            st = str(t.get("source_type", "")).strip().capitalize()
            tt = str(t.get("target_type", "")).strip().capitalize()
            s_raw = norm_space(t.get("source_raw", ""))
            t_raw = norm_space(t.get("target_raw", ""))

            if not s_raw or not t_raw or s_raw.lower() == t_raw.lower():
                continue
            if rel not in ALLOWED_RELATIONS:
                continue
            if st not in ALLOWED_NODE_TYPES:
                st = "Company"
            if tt not in ALLOWED_NODE_TYPES:
                tt = "Company"

            triples_out.append({
                "source_chunk_id": cid,
                "published_date": pub_date,
                "source_raw": s_raw,
                "source_type": st,
                "relation": rel,
                "target_raw": t_raw,
                "target_type": tt,
                "evidence": norm_space(t.get("evidence", row_dict["resolved_text"][:120])),
                "confidence": float(t.get("confidence", 0.9))
            })
    except Exception:
        pass
    return triples_out

def run_extraction(resolved_df, max_workers=10):
    print(f"\n--- Running Triple Extraction (NER + RE) on {len(resolved_df)} chunks (Parallel Workers: {max_workers}) ---", flush=True)
    rows_list = resolved_df.to_dict(orient="records")
    all_triples = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_single_chunk, r) for r in rows_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting Triples"):
            all_triples.extend(f.result())

    raw_triples_df = pd.DataFrame(all_triples)
    print(f"✅ Extracted {len(raw_triples_df)} raw triples.", flush=True)
    return raw_triples_df

# ==========================================
# 6. ENTITY RESOLUTION & DISJOINT-SET (MODULE 3)
# ==========================================
MANUAL_ALIASES = {
    "google": "Google", "google llc": "Google", "alphabet": "Google", "alphabet inc": "Google",
    "meta": "Meta", "meta platforms": "Meta", "facebook": "Meta",
    "microsoft": "Microsoft", "microsoft corp": "Microsoft", "microsoft corporation": "Microsoft",
    "apple": "Apple", "apple inc": "Apple",
    "openai": "OpenAI", "openai inc": "OpenAI",
    "amazon": "Amazon", "amazon.com": "Amazon",
    "ericsson": "Ericsson", "telefonaktiebolaget lm ericsson": "Ericsson",
    "aeris": "Aeris", "aeris communications": "Aeris", "aeris communications inc": "Aeris"
}

LEGAL_SUFFIXES = [
    r"\binc\.?\b", r"\bcorp\.?\b", r"\bcorporation\b", r"\bllc\b",
    r"\bltd\.?\b", r"\btechnologies\b", r"\btech\b", r"\bgroup\b",
    r"\bco\.?\b", r"\bcompany\b", r"\bholdings\b", r"\bplatforms\b"
]

def norm_entity(name):
    return norm_space(name).lower()

def strip_suffix(name):
    s = norm_entity(name)
    for pat in LEGAL_SUFFIXES:
        s = re.sub(pat, "", s, flags=re.I)
    return norm_space(s)

def merge_guard(a, b):
    if a == b:
        return True, "EXACT_NORM"
    if MANUAL_ALIASES.get(a) and MANUAL_ALIASES.get(a) == MANUAL_ALIASES.get(b):
        return True, "MANUAL_ALIAS"

    sa = strip_suffix(a)
    sb = strip_suffix(b)
    if sa and sa == sb:
        return True, "SUFFIX_STRIP"

    words_a = a.split()
    words_b = b.split()
    if len(words_a) == 2 and len(words_b) == 2:
        if words_a[1] == words_b[1] and words_a[0] != words_b[0]:
            return False, "DIFFERENT_FIRST_NAME"

    if (a.startswith(b + " ") or b.startswith(a + " ")):
        sub = a.replace(b, "").strip() if len(a) > len(b) else b.replace(a, "").strip()
        if sub in ["watch", "music", "pay", "tv", "cloud", "ventures", "search", "maps", "health", "analytics", "ai"]:
            return False, "COMPANY_VS_PRODUCT"

    if len(words_a) == 1 and len(words_b) == 1 and a != b:
        return False, "DISTINCT_SINGLE_WORD"

    return True, "GUARD_PASS"

class DisjointSet:
    def __init__(self):
        self.parent = {}
    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def build_entity_resolution(raw_triples_df, embed_model, sim_threshold=0.88):
    print("\n--- Running Entity Resolution (Vector ANN + Lexical Guard + Disjoint-Set) ---", flush=True)
    entities = defaultdict(set)
    for _, r in raw_triples_df.iterrows():
        entities[(r.source_type, norm_entity(r.source_raw))].add(r.source_raw)
        entities[(r.target_type, norm_entity(r.target_raw))].add(r.target_raw)

    audit_records = []
    uf = DisjointSet()
    
    for etype in ["Company", "Person", "Technology"]:
        type_keys = [k for k in entities.keys() if k[0] == etype]
        norm_names = [k[1] for k in type_keys]
        if not norm_names:
            continue

        embs = embed_model.encode(norm_names, normalize_embeddings=True, show_progress_bar=False)
        sim_matrix = np.dot(embs, embs.T)

        for i in range(len(norm_names)):
            name_i = norm_names[i]
            for j in range(i + 1, len(norm_names)):
                name_j = norm_names[j]
                sim = float(sim_matrix[i, j])

                if MANUAL_ALIASES.get(name_i) and MANUAL_ALIASES.get(name_i) == MANUAL_ALIASES.get(name_j):
                    uf.union((etype, name_i), (etype, name_j))
                    audit_records.append({
                        "type": etype, "entity_a": name_i, "entity_b": name_j,
                        "similarity": sim, "decision": "MERGE_MANUAL", "reason": "MANUAL_ALIAS"
                    })
                    continue

                sa = strip_suffix(name_i)
                sb = strip_suffix(name_j)
                if sa and sa == sb and len(sa) > 2:
                    uf.union((etype, name_i), (etype, name_j))
                    audit_records.append({
                        "type": etype, "entity_a": name_i, "entity_b": name_j,
                        "similarity": sim, "decision": "MERGE_SUFFIX", "reason": "IDENTICAL_BASE_NAME"
                    })
                    continue

                if sim >= sim_threshold:
                    passed, reason = merge_guard(name_i, name_j)
                    if passed:
                        uf.union((etype, name_i), (etype, name_j))
                        audit_records.append({
                            "type": etype, "entity_a": name_i, "entity_b": name_j,
                            "similarity": round(sim, 4), "decision": "MERGE_VECTOR", "reason": f"HIGH_SIM_{reason}"
                        })
                    else:
                        audit_records.append({
                            "type": etype, "entity_a": name_i, "entity_b": name_j,
                            "similarity": round(sim, 4), "decision": "REJECT_GUARD", "reason": reason
                        })

    clusters = defaultdict(list)
    for (etype, nname), raws in entities.items():
        root = uf.find((etype, nname))
        clusters[root].extend(list(raws))

    canonical_map = {}
    for root, raw_list in clusters.items():
        etype, root_norm = root
        if MANUAL_ALIASES.get(root_norm):
            canon_name = MANUAL_ALIASES[root_norm]
        else:
            canon_name = Counter(raw_list).most_common(1)[0][0]

        for (e_t, nname) in entities.keys():
            if uf.find((e_t, nname)) == root:
                canonical_map[(e_t, nname)] = canon_name

    audit_df = pd.DataFrame(audit_records)
    print(f"✅ Entity Resolution completed. Audited {len(audit_df)} pair comparisons.", flush=True)
    return canonical_map, audit_df

def canonicalize_triples(raw_df, canonical_map):
    df = raw_df.copy()
    def get_canon(typ, raw):
        norm = norm_entity(raw)
        return canonical_map.get((typ, norm), raw)

    df["source_name"] = [get_canon(t, r) for t, r in zip(df.source_type, df.source_raw)]
    df["target_name"] = [get_canon(t, r) for t, r in zip(df.target_type, df.target_raw)]
    df["source_name_norm"] = df.source_name.map(norm_entity)
    df["target_name_norm"] = df.target_name.map(norm_entity)
    df["source_id"] = [sha1(f"{t}:{n}")[:24] for t, n in zip(df.source_type, df.source_name_norm)]
    df["target_id"] = [sha1(f"{t}:{n}")[:24] for t, n in zip(df.target_type, df.target_name_norm)]
    df = df[df.source_id != df.target_id].reset_index(drop=True)
    return df

# ==========================================
# 7. BULK INGESTION TO NEO4J
# ==========================================
def build_nodes_table(triples_df):
    node_records = {}
    for _, r in triples_df.iterrows():
        sid = r.source_id
        if sid not in node_records:
            node_records[sid] = {
                "id": sid, "name": r.source_name, "name_norm": r.source_name_norm,
                "type": r.source_type, "aliases": [r.source_raw]
            }
        else:
            if r.source_raw not in node_records[sid]["aliases"]:
                node_records[sid]["aliases"].append(r.source_raw)

        tid = r.target_id
        if tid not in node_records:
            node_records[tid] = {
                "id": tid, "name": r.target_name, "name_norm": r.target_name_norm,
                "type": r.target_type, "aliases": [r.target_raw]
            }
        else:
            if r.target_raw not in node_records[tid]["aliases"]:
                node_records[tid]["aliases"].append(r.target_raw)

    return pd.DataFrame(list(node_records.values()))

def bulk_ingest_neo4j(nodes_df, triples_df, batch_size=1000):
    print(f"\n--- Ingesting {len(nodes_df)} Nodes and {len(triples_df)} Edges into Neo4j ---", flush=True)
    node_list = nodes_df.to_dict(orient="records")
    for i in range(0, len(node_list), batch_size):
        b = node_list[i:i+batch_size]
        run_cypher("""
        UNWIND $rows AS row
        MERGE (n:Entity {id: row.id})
        SET n.name = row.name,
            n.name_norm = row.name_norm,
            n.type = row.type,
            n.aliases = row.aliases
        """, rows=b)

    for rel_type in ALLOWED_RELATIONS:
        sub = triples_df[triples_df.relation == rel_type]
        if sub.empty:
            continue
        edge_list = sub.to_dict(orient="records")
        for i in range(0, len(edge_list), batch_size):
            b = edge_list[i:i+batch_size]
            cql = f"""
            UNWIND $rows AS row
            MATCH (s:Entity {{id: row.source_id}})
            MATCH (t:Entity {{id: row.target_id}})
            MERGE (s)-[r:{rel_type} {{source_chunk_id: row.source_chunk_id}}]->(t)
            SET r.published_date = row.published_date,
                r.evidence = row.evidence,
                r.confidence = row.confidence
            """
            run_cypher(cql, rows=b)

    print("✅ Neo4j Bulk Ingestion completed.", flush=True)

def verify_graph_integrity():
    print("\n--- Verifying Graph Integrity & Provenance ---", flush=True)
    nodes_cnt = run_cypher("MATCH (n:Entity) RETURN count(n) AS cnt")[0]["cnt"]
    edges_cnt = run_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")[0]["cnt"]
    missing_prov = run_cypher("""
    MATCH ()-[r]->()
    WHERE r.source_chunk_id IS NULL OR r.published_date IS NULL
    RETURN count(r) AS cnt
    """)[0]["cnt"]

    print(f"Total Nodes: {nodes_cnt}", flush=True)
    print(f"Total Edges: {edges_cnt}", flush=True)
    print(f"Invalid / Missing Provenance Edges: {missing_prov}", flush=True)
    assert missing_prov == 0, f"Error: {missing_prov} edges missing provenance!"
    print("✅ Provenance Integrity Check: 100% PASS (0 invalid edges).", flush=True)
    return {"nodes": nodes_cnt, "edges": edges_cnt, "invalid_provenance": missing_prov}

# ==========================================
# 8. RETRIEVAL ARCHITECTURE (MODULE 4)
# ==========================================
class FlatRAGRetriever:
    def __init__(self, chunks_df, embed_model):
        self.chunks_df = chunks_df.reset_index(drop=True)
        self.embed_model = embed_model
        print(f"Building FAISS Index for {len(chunks_df)} chunks...", flush=True)
        texts = chunks_df["text"].tolist()
        embs = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs.astype(np.float32))
        print("✅ FAISS Index ready.", flush=True)

    def retrieve(self, query, k=6):
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb.astype(np.float32), k)
        results = []
        for rank, idx in enumerate(idxs[0]):
            if idx >= 0 and idx < len(self.chunks_df):
                r = self.chunks_df.iloc[idx]
                results.append({
                    "rank": rank + 1, "score": float(scores[0][rank]),
                    "chunk_id": r.chunk_id, "published_at": r.published_at,
                    "title": r.title, "text": r.text
                })
        context = "\n\n".join([f"[{res['chunk_id']} | {res['published_at']}] {res['text']}" for res in results])
        return context, results

class GraphRAGRetriever:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        nodes = run_cypher("MATCH (n:Entity) RETURN n.id AS id, n.name AS name, n.name_norm AS name_norm, n.type AS type, n.aliases AS aliases")
        self.nodes_df = pd.DataFrame(nodes)
        if not self.nodes_df.empty:
            self.node_embs = embed_model.encode(self.nodes_df["name_norm"].tolist(), normalize_embeddings=True, show_progress_bar=False)
        else:
            self.node_embs = None

    def match_seeds(self, query, fuzzy_threshold=0.66):
        prompt = f"Extract 1 to 4 key Named Entities from this question for Knowledge Graph search:\nQUESTION: {query}\nReturn JSON: {{\"seeds\": [\"Entity1\", \"Entity2\"]}}"
        try:
            res, _ = groq_json("You are an entity extractor. Return JSON: {\"seeds\": [\"...\"]}", prompt)
            raw_seeds = res.get("seeds", [])
        except:
            raw_seeds = [query]

        matched_node_ids = set()
        matched_details = []

        for s in raw_seeds:
            s_norm = norm_entity(s)
            exact = self.nodes_df[self.nodes_df.name_norm == s_norm] if not self.nodes_df.empty else pd.DataFrame()
            if not exact.empty:
                for _, r in exact.iterrows():
                    matched_node_ids.add(r.id)
                    matched_details.append({"seed": s, "node_id": r.id, "name": r.name, "method": "EXACT"})
                continue

            if self.node_embs is not None and len(self.node_embs) > 0:
                s_emb = self.embed_model.encode([s_norm], normalize_embeddings=True)
                sims = np.dot(self.node_embs, s_emb.T).flatten()
                best_idx = np.argmax(sims)
                if sims[best_idx] >= fuzzy_threshold:
                    r = self.nodes_df.iloc[best_idx]
                    matched_node_ids.add(r.id)
                    matched_details.append({"seed": s, "node_id": r.id, "name": r.name, "sim": float(sims[best_idx]), "method": "FUZZY_VECTOR"})

        return list(matched_node_ids), matched_details

    def retrieve_subgraph(self, seed_node_ids, max_hops=2, limit_per_node=50):
        if not seed_node_ids:
            return {"context": "", "edges": [], "supernode_events": []}

        visited_nodes = set(seed_node_ids)
        frontier = deque(seed_node_ids)
        all_edges = []
        supernode_events = []

        for hop in range(max_hops):
            next_frontier = deque()
            while frontier:
                curr_node = frontier.popleft()
                deg_res = run_cypher("MATCH (n:Entity {id: $id})-[r]-() RETURN count(r) AS degree", id=curr_node)
                degree = deg_res[0]["degree"] if deg_res else 0

                fetch_limit = limit_per_node
                if degree > SUPER_NODE_DEGREE:
                    supernode_events.append({"node_id": curr_node, "degree": degree, "capped_at": limit_per_node})

                edge_query = """
                MATCH (s:Entity {id: $id})-[r]->(t:Entity)
                RETURN s.id AS s_id, s.name AS s_name, s.type AS s_type,
                       type(r) AS relation, r.published_date AS published_date,
                       r.source_chunk_id AS source_chunk_id, r.evidence AS evidence,
                       t.id AS t_id, t.name AS t_name, t.type AS t_type
                ORDER BY r.published_date DESC
                LIMIT $limit
                """
                edges = run_cypher(edge_query, id=curr_node, limit=fetch_limit)
                for e in edges:
                    all_edges.append(e)
                    if e["t_id"] not in visited_nodes:
                        visited_nodes.add(e["t_id"])
                        next_frontier.append(e["t_id"])

                if len(all_edges) >= GLOBAL_EDGE_CAP:
                    break
            frontier = next_frontier
            if len(all_edges) >= GLOBAL_EDGE_CAP:
                break

        lines = []
        for e in all_edges[:GLOBAL_EDGE_CAP]:
            line = f"{e['s_name']} [{e['s_type']}] -{e['relation']}-> {e['t_name']} [{e['t_type']}] | date={e.get('published_date','')} | chunk={e.get('source_chunk_id','')} | evidence={e.get('evidence','')}"
            lines.append(line)

        graph_context = "\n".join(lines)[:MAX_GRAPH_CONTEXT_CHARS]
        return {"context": graph_context, "edges": all_edges, "supernode_events": supernode_events}

# ==========================================
# 9. ANSWER GENERATION (FLAT VS GRAPH)
# ==========================================
RAG_SYSTEM = """
You are an expert AI business intelligence assistant.
Answer the user's question accurately, completely, and faithfully based ONLY on the provided CONTEXT.
If information is missing, clearly state what is unknown. Include specific company names, dates, technologies, and provenance where relevant.
""".strip()

def answer_flat_rag(question, flat_retriever):
    t0 = time.time()
    context, results = flat_retriever.retrieve(question, k=6)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    ans, usage = groq_chat([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": user_prompt}
    ])
    lat = round(time.time() - t0, 3)
    return {
        "answer": ans, "context": context, "latency_s": lat,
        "total_tokens": usage.get("total_tokens", 0)
    }

def answer_graph_rag(question, graph_retriever, flat_retriever):
    t0 = time.time()
    seeds, seed_details = graph_retriever.match_seeds(question)
    graph_res = graph_retriever.retrieve_subgraph(seeds, max_hops=2, limit_per_node=50)
    flat_context, _ = flat_retriever.retrieve(question, k=3)

    hybrid_context = f"=== KNOWLEDGE GRAPH CONTEXT ===\n{graph_res['context']}\n\n=== RELEVANT CHUNKS ===\n{flat_context}"
    user_prompt = f"CONTEXT:\n{hybrid_context}\n\nQUESTION: {question}"
    ans, usage = groq_chat([
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": user_prompt}
    ])
    lat = round(time.time() - t0, 3)
    return {
        "answer": ans, "context": hybrid_context, "latency_s": lat,
        "total_tokens": usage.get("total_tokens", 0),
        "graph_debug": {"seeds": seed_details, "diagnostics": graph_res}
    }

# ==========================================
# 10. LLM-AS-A-JUDGE & EVALUATION (MODULE 5)
# ==========================================
JUDGE_SYSTEM = """
You are a rigorous, impartial LLM Judge evaluating RAG systems.
Evaluate candidate answers on 3 criteria (Score 1 to 5):
1. Comprehensiveness (1-5): Does the answer cover all key entities, facts, and relationships asked?
2. Faithfulness (1-5): Is every statement directly supported by the candidate context without hallucinations?
3. Multi-hop Reasoning (1-5): Did the system correctly chain together connections across multiple facts/documents?

Return strict JSON format:
{
  "comprehensiveness": 5,
  "faithfulness": 5,
  "multi_hop_reasoning": 5,
  "rationale": "2-4 sentences explaining the score"
}
""".strip()

def judge_answer(question, reference, answer, context):
    prompt = f"""
QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{answer}

CANDIDATE CONTEXT:
{context[:16000]}
"""
    try:
        obj, _ = groq_json(JUDGE_SYSTEM, prompt, model=JUDGE_MODEL)
        out = {}
        for k in ["comprehensiveness", "faithfulness", "multi_hop_reasoning"]:
            out[k] = max(1, min(5, int(obj.get(k, 3))))
        out["rationale"] = norm_space(obj.get("rationale", "Evaluated faithfully."))
        return out
    except Exception as e:
        return {"comprehensiveness": 3, "faithfulness": 3, "multi_hop_reasoning": 3, "rationale": str(e)}

def evaluate_single_question(q, flat_retriever, graph_retriever):
    flat = answer_flat_rag(q.question, flat_retriever)
    graph = answer_graph_rag(q.question, graph_retriever, flat_retriever)

    jf = judge_answer(q.question, q.reference_answer, flat["answer"], flat["context"])
    jg = judge_answer(q.question, q.reference_answer, graph["answer"], graph["context"])

    return {
        "id": q.id, "group": q.group, "question": q.question,
        "reference_answer": q.reference_answer,
        "flat_answer": flat["answer"], "graph_answer": graph["answer"],
        "flat_comprehensiveness": jf["comprehensiveness"],
        "graph_comprehensiveness": jg["comprehensiveness"],
        "flat_faithfulness": jf["faithfulness"],
        "graph_faithfulness": jg["faithfulness"],
        "flat_multi_hop_reasoning": jf["multi_hop_reasoning"],
        "graph_multi_hop_reasoning": jg["multi_hop_reasoning"],
        "flat_latency_s": flat["latency_s"],
        "graph_latency_s": graph["latency_s"],
        "flat_total_tokens": flat.get("total_tokens", 0),
        "graph_total_tokens": graph.get("total_tokens", 0),
        "flat_judge_rationale": jf["rationale"],
        "graph_judge_rationale": jg["rationale"],
        "graph_supernode_events": len(graph["graph_debug"]["diagnostics"].get("supernode_events", []))
    }

def run_evaluation(golden_df, flat_retriever, graph_retriever, max_eval_samples=15, max_workers=5):
    print(f"\n--- Running Evaluation on {min(len(golden_df), max_eval_samples)} Golden Questions (Parallel Workers: {max_workers}) ---", flush=True)
    eval_subset = golden_df.head(max_eval_samples).copy()
    q_list = list(eval_subset.itertuples(index=False))
    rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_single_question, q, flat_retriever, graph_retriever): q for q in q_list}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Qs"):
            rows.append(f.result())

    eval_results_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    eval_results_df.to_csv("outputs/graphrag_eval_results.csv", index=False)
    print("✅ Saved outputs/graphrag_eval_results.csv", flush=True)
    return eval_results_df

def generate_comparison_table(eval_df):
    metric_map = {
        "Comprehensiveness": ("flat_comprehensiveness", "graph_comprehensiveness"),
        "Faithfulness": ("flat_faithfulness", "graph_faithfulness"),
        "Multi-hop reasoning": ("flat_multi_hop_reasoning", "graph_multi_hop_reasoning"),
        "Latency (s)": ("flat_latency_s", "graph_latency_s"),
        "Token usage": ("flat_total_tokens", "graph_total_tokens"),
    }

    rows = []
    for group, g in eval_df.groupby("group"):
        for metric, (fc, gc) in metric_map.items():
            f_val = pd.to_numeric(g[fc], errors="coerce").mean()
            g_val = pd.to_numeric(g[gc], errors="coerce").mean()

            if metric in {"Latency (s)", "Token usage"}:
                comment = "Flat RAG nhanh hơn / ít token hơn." if f_val < g_val else "Chi phí token tương đương."
            else:
                delta = g_val - f_val
                if delta >= 0.5:
                    comment = "GraphRAG vượt trội rõ rệt nhờ liên kết tri thức xuyên văn bản."
                elif delta <= -0.5:
                    comment = "Flat RAG tốt hơn; đồ thị có thể bị thiếu seed hoặc nhiễu."
                else:
                    comment = "Hai phương pháp có hiệu quả tương đương."

            rows.append({
                "Loại câu hỏi": group, "Metric": metric,
                "Flat RAG": round(f_val, 3) if pd.notna(f_val) else np.nan,
                "GraphRAG": round(g_val, 3) if pd.notna(g_val) else np.nan,
                "Nhận xét phân tích": comment
            })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv("outputs/graphrag_vs_flatrag_summary.csv", index=False)
    print("✅ Saved outputs/graphrag_vs_flatrag_summary.csv", flush=True)
    return comp_df

# ==========================================
# 11. BONUS: COMMUNITY DETECTION & SELF-CORRECTION
# ==========================================
def run_community_detection():
    print("\n--- Running Bonus: NetworkX Community Detection ---", flush=True)
    edges_raw = run_cypher("MATCH (a:Entity)-[r]->(b:Entity) RETURN a.id AS source, b.id AS target LIMIT 20000")
    if not edges_raw:
        print("No edges for community detection.", flush=True)
        return pd.DataFrame()
    edge_df = pd.DataFrame(edges_raw)
    G = nx.Graph()
    G.add_edges_from(edge_df[["source", "target"]].itertuples(index=False, name=None))
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    print(f"Discovered {len(communities)} distinct communities.", flush=True)

    rows = []
    for cid, members in enumerate(communities):
        for node_id in members:
            rows.append({"id": node_id, "community_id": int(cid)})

    comm_df = pd.DataFrame(rows)
    for i in range(0, len(rows), 1000):
        b = rows[i:i+1000]
        run_cypher("""
        UNWIND $rows AS row
        MATCH (n:Entity {id: row.id})
        SET n.community_id = row.community_id
        """, rows=b)
    print("✅ Community IDs written to Neo4j nodes.", flush=True)
    return comm_df

print("\nPipeline script loaded and ready to execute.", flush=True)
