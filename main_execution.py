import os, sys, time, json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import run_full_pipeline as pipe

def main():
    print("=" * 70)
    print("🚀 STARTING FULL LAB 19 GRAPHRAG PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Setup Neo4j schema
    pipe.setup_graph_schema()

    # 2. Load articles & Build Chunks
    print(f"\nLoading articles from {pipe.DATA_PATH}...")
    df_raw = pd.read_csv(pipe.DATA_PATH)
    print(f"Loaded {len(df_raw)} articles.")

    # Exact dedup by sha1(title + description)
    df_raw["dedup_key"] = [pipe.sha1(f"{pipe.norm_space(t)} {pipe.norm_space(d)}") for t, d in zip(df_raw.get("title", ""), df_raw.get("description", ""))]
    df_articles = df_raw.drop_duplicates(subset=["dedup_key"]).reset_index(drop=True)
    print(f"After exact dedup: {len(df_articles)} articles (removed {len(df_raw) - len(df_articles)} duplicates).")

    chunks_df = pipe.build_chunks(df_articles, max_chunks=pipe.LAB_MAX_CHUNKS)
    print(f"Generated {len(chunks_df)} chunks (avg words: {chunks_df['word_count'].mean():.1f}).")

    # 3. Load Embedding Model
    print("\nLoading sentence transformer model 'all-MiniLM-L6-v2'...")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 4. Conservative Coreference Resolution (with caching)
    cache_coref = "outputs/resolved_chunks.csv"
    if os.path.exists(cache_coref):
        print(f"Loading cached resolved chunks from {cache_coref}...")
        resolved_df = pd.read_csv(cache_coref)
        unresolved_mentions = []
    else:
        resolved_df, unresolved_mentions = pipe.run_coreference_resolution(
            chunks_df, max_chunks=pipe.EXTRACTION_MAX_CHUNKS, batch_size=5, max_workers=8
        )
        resolved_df.to_csv(cache_coref, index=False)

    # 5. Triple Extraction (NER + RE)
    cache_triples = "outputs/extracted_triples.csv"
    if os.path.exists(cache_triples):
        print(f"Loading cached raw triples from {cache_triples}...")
        raw_triples_df = pd.read_csv(cache_triples)
    else:
        raw_triples_df = pipe.run_extraction(resolved_df, max_workers=10)
        raw_triples_df.to_csv(cache_triples, index=False)

    # 6. Entity Resolution with Vector ANN + Lexical Guard + Disjoint-Set
    canonical_map, audit_df = pipe.build_entity_resolution(raw_triples_df, embed_model, sim_threshold=0.88)
    triples_df = pipe.canonicalize_triples(raw_triples_df, canonical_map)
    nodes_df = pipe.build_nodes_table(triples_df)

    print(f"\nCanonical triples count: {len(triples_df)} (from {len(raw_triples_df)} raw triples)")
    print(f"Unique canonical entities: {len(nodes_df)}")

    # 7. Bulk Ingestion into Neo4j
    pipe.bulk_ingest_neo4j(nodes_df, triples_df, batch_size=1000)
    integrity = pipe.verify_graph_integrity()

    # 8. Super-node & Graph Diagnostics
    print("\n--- Super-node Diagnostics ---")
    top_supernodes = pipe.run_cypher("""
    MATCH (n:Entity)-[r]-()
    WITH n, count(r) AS degree
    ORDER BY degree DESC LIMIT 5
    RETURN n.id AS id, n.name AS name, n.type AS type, degree
    """)
    print("Top 5 Super-nodes in Graph:")
    for sn in top_supernodes:
        print(f" - [{sn['type']}] {sn['name']} (degree: {sn['degree']})")

    # Test supernode policy cap
    if top_supernodes:
        sn_top = top_supernodes[0]
        deg = sn_top["degree"]
        lim = 50 if deg > pipe.SUPER_NODE_DEGREE else 1000
        edges_sn = pipe.run_cypher("""
        MATCH (s:Entity {id: $id})-[r]->(t:Entity)
        RETURN s.name, type(r), t.name, r.published_date
        ORDER BY r.published_date DESC LIMIT $lim
        """, id=sn_top["id"], lim=lim)
        print(f"Super-node cap test on '{sn_top['name']}': Degree={deg}, Fetched={len(edges_sn)} (Cap Limit={lim})")

    # 9. Bonus: Community Detection
    comm_df = pipe.run_community_detection()

    # 10. Retrievers Initialization
    flat_retriever = pipe.FlatRAGRetriever(chunks_df, embed_model)
    graph_retriever = pipe.GraphRAGRetriever(embed_model)

    # 11. Golden Dataset Evaluation
    golden_file = "data/graphrag_golden_50_first5000.csv"
    if os.path.exists(golden_file):
        golden_df = pd.read_csv(golden_file)
        print(f"\nLoaded Golden Dataset from {golden_file} with {len(golden_df)} questions.")
    else:
        golden_df = pd.DataFrame([
            {"id": "G01", "group": "factoid", "question": "Who is the CEO of Hugging Face in 2023?", "reference_answer": "Clément Delangue"},
            {"id": "G02", "group": "multi-hop", "question": "What companies acquired IoT business from Ericsson?", "reference_answer": "Aeris acquired Ericsson IoT Accelerator and Connected Vehicle Cloud."},
            {"id": "G03", "group": "cross-doc", "question": "Compare the IoT market reach of Aeris before and after the Ericsson acquisition.", "reference_answer": "Aeris expanded to 100M+ devices across 190 countries and 9000 enterprises."}
        ])

    eval_results_df = pipe.run_evaluation(golden_df, flat_retriever, graph_retriever, max_eval_samples=15, max_workers=5)
    summary_df = pipe.generate_comparison_table(eval_results_df)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY (FLAT RAG VS GRAPHRAG)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Save diagnostic summary for lab report
    diag = {
        "integrity": integrity,
        "top_supernodes": top_supernodes,
        "audit_samples": audit_df.head(20).to_dict(orient="records") if not audit_df.empty else [],
        "rejected_guards": audit_df[audit_df["decision"] == "REJECT_GUARD"].head(10).to_dict(orient="records") if not audit_df.empty else [],
        "unresolved_mentions_sample": unresolved_mentions[:10],
        "eval_summary": summary_df.to_dict(orient="records")
    }
    with open("outputs/diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    print("\n✅ ALL PIPELINE MODULES COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
