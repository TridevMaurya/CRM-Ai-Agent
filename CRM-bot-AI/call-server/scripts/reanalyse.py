"""Re-run the post-call analysis for every finished call (e.g. after the analysis
schema changed) and re-bucket. Usage: python scripts/reanalyse.py [max_calls]"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import agents, analysis, store  # noqa: E402


async def main(limit: int):
    store.init()
    calls = [c for c in store.list_calls(limit=limit) if c["status"] != "live"]
    done = 0
    for c in calls:
        full = store.get_call(c["id"])
        agent = agents.get_agent(full["agent_id"])
        if not agent or not full["turns"]:
            continue
        result = await analysis.analyse_call(agent, full["turns"])
        if result and "error" not in result:
            store.set_analysis(c["id"], result); done += 1
            print(c["id"][:8], store.bucket_of(result), (result.get("summary") or "")[:70])
    print(f"re-analysed {done} of {len(calls)}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000))
