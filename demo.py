"""Run: python demo.py "your research topic" """
import sys

from agentcrew.graph import run_crew

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "Retrieval-Augmented Generation for enterprise Q&A"
    result = run_crew(topic)

    print("=" * 70)
    print("AGENT TRACE")
    print("=" * 70)
    for line in result["trace"]:
        print(" -", line)

    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result["draft"])

    print()
    print("=" * 70)
    print(f"Revisions: {result['revision_count']} | Approved: {result.get('approved', False)}")
