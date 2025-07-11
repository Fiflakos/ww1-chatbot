# test_retrieval.py

from agents.retrieval_agent import RetrievalAgent


def main():
    # 1) point at your cleaned corpus
    retriever = RetrievalAgent(corpus_dir="data_cleaned")

    # 2) ask anything
    query = input("🔍 Enter your query: ").strip()
    hits  = retriever.search(query, top_k=5)

    # 3) print out the top passages
    if not hits:
        print("No relevant passages found.")
    else:
        for i, (fn, score, snippet) in enumerate(hits, start=1):
            print(f"{i}. {fn}  (score {score:.2f})")
            print(f"   → {snippet}…\n")

if __name__ == "__main__":
    main()
