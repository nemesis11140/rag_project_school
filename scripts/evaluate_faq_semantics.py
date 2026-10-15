"""使用现有 LLM 实测 FAQ 问题等价判断，需显式加 --live；不访问数据库。"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mysql_qa.semantic_matcher import SemanticQAMatcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="调用 .env 配置的大模型 API")
    parser.add_argument("--output", default="runtime/faq-semantics-evaluation.json")
    args = parser.parse_args()
    if not args.live:
        parser.error("真实 API 验证请添加 --live；离线回归使用 tests/test_faq_semantics.py")
    matcher = SemanticQAMatcher()
    if not matcher.api_url or not matcher.api_key:
        parser.error("请先配置 LLM_API_KEY 和 API 地址")
    cases = json.loads((ROOT / "tests/fixtures/faq_semantic_cases.json").read_text(encoding="utf-8"))

    def evaluate(case):
        started = time.perf_counter()
        match = matcher.select_match(case["query"], [{"question": case["candidate"], "id": 0}])
        result = {**case, "accepted": match is not None,
                  "confidence": match.get("semantic_confidence") if match else None,
                  "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
        result["passed"] = result["accepted"] == case["equivalent"]
        print(f"{'PASS' if result['passed'] else 'FAIL'} {case['query']} -> {case['candidate']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(evaluate, cases))
    report = {"model": matcher.model_name, "policy_key": matcher.policy_key,
              "passed": sum(item["passed"] for item in results), "total": len(results),
              "false_accepts": sum(item["accepted"] and not item["equivalent"] for item in results),
              "false_rejects": sum(not item["accepted"] and item["equivalent"] for item in results),
              "cases": results}
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"通过 {report['passed']}/{report['total']}；误接收 {report['false_accepts']}；报告 {path}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
