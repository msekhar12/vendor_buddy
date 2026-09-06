"""
CIVSA evaluation harness.

Runs a fixed query set through the pipeline (gate → qa) and scores each
result against expected route + answer keywords. Writes a timestamped
JSON summary to eval_results/ and prints a per-family pass table.

Usage:
    python -m scripts.eval                       # run all
    python -m scripts.eval --family sql          # run one family
    python -m scripts.eval --id iso-9001-list    # run one query
    python -m scripts.eval --http                # use /api/query instead of in-process
    python -m scripts.eval --verbose             # print full answer on FAIL
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
QUERIES_PATH = ROOT / "scripts" / "eval_queries.json"
RESULTS_DIR  = ROOT / "eval_results"


# ------------------------------------------------------------------
# Runners
# ------------------------------------------------------------------

def run_query_inproc(query: str, user: str = "eval") -> dict:
    """Call gate + qa directly. Fast, no server required."""
    from civsa.gate.pipeline import process as gate_process
    from civsa.qa import answer as qa_answer

    g = gate_process(query, user=user)
    if not g.get("allowed", False):
        return {
            "allowed":      False,
            "route_method": f"gate_{g.get('reason', 'blocked')}",
            "answer":       g.get("message", ""),
            "sources":      [],
        }
    return {"allowed": True, **qa_answer(query)}


def run_query_http(query: str, user: str = "eval") -> dict:
    """Hit /api/query over HTTP. Requires FastAPI running."""
    import requests
    r = requests.post(
        "http://localhost:8000/api/query",
        data={"q": query, "user": user},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def score(expected: dict, actual: dict) -> tuple[bool, list[str]]:
    """
    Compare actual vs expected on: route, allowed, and three keyword lists:
      must_contain_all — every keyword must appear (case-insensitive)
      must_contain_any — at least one must appear
      must_not_contain — none must appear
    """
    reasons: list[str] = []
    answer_lower = str(actual.get("answer", "")).lower()

    exp_route = expected.get("expected_route")
    if exp_route:
        act_route = actual.get("route_method", "")
        if exp_route not in act_route:
            reasons.append(f"route: expected '{exp_route}', got '{act_route}'")

    if "expected_allowed" in expected:
        exp_ok = bool(expected["expected_allowed"])
        act_ok = bool(actual.get("allowed", False))
        if exp_ok != act_ok:
            reasons.append(f"allowed: expected {exp_ok}, got {act_ok}")

    for kw in expected.get("must_contain_all", []):
        if kw.lower() not in answer_lower:
            reasons.append(f"missing keyword: '{kw}'")

    any_kws = expected.get("must_contain_any", [])
    if any_kws and not any(kw.lower() in answer_lower for kw in any_kws):
        reasons.append(f"missing any of: {any_kws}")

    for kw in expected.get("must_not_contain", []):
        if kw.lower() in answer_lower:
            reasons.append(f"unexpected keyword: '{kw}'")

    return (len(reasons) == 0, reasons)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="CIVSA eval harness")
    p.add_argument("--family", help="run only one family")
    p.add_argument("--id",     help="run only one query by id")
    p.add_argument("--http",   action="store_true",
                   help="use /api/query instead of in-process")
    p.add_argument("--verbose", action="store_true",
                   help="print full answer text on FAIL")
    args = p.parse_args()

    if not QUERIES_PATH.exists():
        print(f"Query file not found: {QUERIES_PATH}", file=sys.stderr)
        sys.exit(1)
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    if args.family:
        queries = [q for q in queries if q.get("family") == args.family]
    if args.id:
        queries = [q for q in queries if q.get("id") == args.id]
    if not queries:
        print("No queries matched filter.")
        sys.exit(0)

    runner = run_query_http if args.http else run_query_inproc
    mode   = "HTTP" if args.http else "in-process"
    print(f"Running {len(queries)} queries via {mode}…\n")

    results: list[dict] = []
    passed = 0
    family_stats: dict[str, dict[str, int]] = {}
    started = time.time()

    for i, q in enumerate(queries, 1):
        qid    = q.get("id", f"q{i}")
        family = q.get("family", "unknown")
        text   = q["query"]
        print(f"[{i:>2}/{len(queries)}] {qid:36s} · {family:8s} · {text[:60]}")

        t0 = time.time()
        try:
            actual = runner(text)
            error = None
        except Exception as e:  # noqa: BLE001
            actual = {"error": str(e)}
            error = str(e)
        elapsed = time.time() - t0

        ok, reasons = (False, [f"exception: {error}"]) if error else score(q, actual)
        if ok:
            passed += 1
            print(f"        PASS  ({elapsed:.1f}s)  route={actual.get('route_method', '?')}")
        else:
            print(f"        FAIL  ({elapsed:.1f}s)  route={actual.get('route_method', '?')}")
            for r in reasons:
                print(f"          · {r}")
            if args.verbose:
                ans = str(actual.get("answer", ""))[:400]
                print(f"          answer: {ans}…" if len(ans) == 400 else f"          answer: {ans}")

        results.append({
            "id": qid, "family": family, "query": text,
            "expected": q, "actual": actual,
            "passed": ok, "reasons": reasons,
            "elapsed_s": round(elapsed, 3),
        })
        stat = family_stats.setdefault(family, {"pass": 0, "fail": 0})
        stat["pass" if ok else "fail"] += 1

    total_elapsed = time.time() - started
    total = len(queries)

    print(f"\n{'=' * 60}")
    print(f"Overall:  {passed}/{total} passed ({100*passed/total:.0f}%)")
    print(f"Wall:     {total_elapsed:.1f}s ({total_elapsed/total:.2f}s/query)")
    print(f"{'=' * 60}")
    print("By family:")
    for family in sorted(family_stats):
        s = family_stats[family]
        n = s["pass"] + s["fail"]
        print(f"  {family:10s}  {s['pass']:2d}/{n:2d}  ({100*s['pass']/n:.0f}%)")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp":  ts,
        "total":      total,
        "passed":     passed,
        "failed":     total - passed,
        "pass_rate":  passed / total,
        "elapsed_s":  round(total_elapsed, 1),
        "by_family":  family_stats,
        "results":    results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved detailed results to {out_path}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()