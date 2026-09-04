"""
Debug utility for inspecting the CIVSA indexes.

Usage:
  PYTHONPATH=. python scripts/inspect_index.py                        # stats
  PYTHONPATH=. python scripts/inspect_index.py "hex bolt ISO"         # query both stores
  PYTHONPATH=. python scripts/inspect_index.py --list-vendors         # show vendors
  PYTHONPATH=. python scripts/inspect_index.py --dump Rajshree_Fasteners  # all chunks for a vendor
"""
import sys

from civsa import tfidf_store, vector_store


def stats():
    n_chroma = vector_store._collection.count()
    tf = tfidf_store._load()
    print(f"Chroma chunks:  {n_chroma}")
    print(f"TF-IDF chunks:  {len(tf['chunks'])}")
    if tf["chunks"]:
        sources = sorted({m["source"] for m in tf["metas"]})
        print(f"Distinct source files: {len(sources)}")
        for s in sources[:10]:
            print(f"  - {s}")


def list_vendors():
    tf = tfidf_store._load()
    vendors = {}
    for m in tf["metas"]:
        vendors.setdefault(m["vendor"], set()).add(m["source"])
    for v, srcs in sorted(vendors.items()):
        print(f"{v}: {len(srcs)} doc(s)")
        for s in sorted(srcs):
            print(f"  - {s}")


def dump_vendor(vendor):
    tf = tfidf_store._load()
    matching = [(c, m) for c, m in zip(tf["chunks"], tf["metas"])
                if m["vendor"] == vendor]
    print(f"Vendor {vendor}: {len(matching)} chunks")
    for c, m in matching[:20]:
        print(f"\n--- {m['source']} para {m['para']} labels={m.get('labels')} ---")
        print(c[:200])


def query_both(text, k=5):
    print(f"\n=== Chroma (vector) — top {k} ===")
    hits = vector_store.query(text, k=k)
    for i, (doc, meta, dist) in enumerate(zip(
            hits["documents"][0], hits["metadatas"][0], hits["distances"][0]), 1): # type: ignore
        print(f"{i}. dist={dist:.3f}  labels={meta['labels']}  {meta['source']}#para{meta['para']}")
        print(f"   {doc[:150].strip()}")

    print(f"\n=== TF-IDF — top {k*2} ===")
    # Inline the scoring to also print numeric scores
    data = tfidf_store._load()
    if not data["chunks"]:
        print("  (empty index)"); return
    vec = data["vectorizer"]
    q = vec.transform([text])
    scores = (data["matrix"] @ q.T).toarray().ravel()
    top = scores.argsort()[::-1][: k * 2]
    for rank, i in enumerate(top, 1):
        if scores[i] <= 0: continue
        m = data["metas"][i]
        print(f"{rank}. tfidf={scores[i]:.3f}  labels={m.get('labels')}  {m['source']}#para{m['para']}")
        print(f"   {data['chunks'][i][:150].strip()}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        stats()
    elif args[0] == "--list-vendors":
        list_vendors()
    elif args[0] == "--dump" and len(args) > 1:
        dump_vendor(args[1])
    else:
        query_both(" ".join(args))