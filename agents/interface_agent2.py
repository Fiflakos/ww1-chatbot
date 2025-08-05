import os
import streamlit as st
import openai
import json
from retrieval_agent import RetrievalAgent

# 1) Set OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2) Load full dataset for source ID → full document preview
@st.cache_data
def load_full_documents(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Build mapping: source_id → full doc
    return {item.get("source", f"doc_{i}"): item for i, item in enumerate(data)}

full_docs = load_full_documents("data/annotated2_ww1_qa.json")

# 3) Load BM25 retriever
@st.cache_resource
def load_retriever():
    return RetrievalAgent(json_path="data/annotated2_ww1_qa.json")

retriever = load_retriever()

st.set_page_config(page_title="🪖 WW1 Historical Chatbot RAG 🪖")
st.title("🪖 WW1 Historical Chatbot RAG 🪖")
st.write("Ask me questions about the First World War")

q = st.text_input("💬 What would you like to know?")
if not q:
    st.stop()

# 4) Retrieve top-k documents
hits = retriever.search(q, top_k=3)

if not hits:
    st.warning("No relevant passages found.")
    st.stop()

# 5) Build prompt from hits
context = "\n\n".join(f"{src}: {snip}" for src, _, snip in hits)
system = "You are a WW1 historian assistant. Answer concisely and cite which letter/diary entry you used."
user = f"Context:\n{context}\n\nQuestion: {q}"
cited_sources = sorted(set(src for src, _, _ in hits))

# 6) Generate answer before showing documents
try:
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.7,
        max_tokens=256,
    )
    answer = resp.choices[0].message.content
    answer += f"\n\n_(sources: {', '.join(cited_sources)})_"
    st.markdown(f"**📜 Answer:** {answer}")
except Exception as e:
    st.error(f"❌ OpenAI API error: {e}")
    st.stop()

# 7) Show passage summaries with full document previews
st.subheader("🔍 Top passages retrieved by Unified_RAG")

for i, (source_id, score, snippet) in enumerate(hits, 1):
    st.markdown(f"**{i}. Source ID: `{source_id}`**")
    st.markdown(f"- **Unified_RAG Score:** `{score:.4f}`")
    st.markdown(f"- **Snippet:** {snippet[:300]}…")

    # Expander for full document preview
    if source_id in full_docs:
        doc = full_docs[source_id]
        with st.expander(f"📄 Click to view full document: {source_id}"):
            st.markdown(f"**Source ID:** `{source_id}`")
            st.markdown(f"**Query:** {doc.get('query', '—')}")
            st.markdown(f"**Response:** {doc.get('response', '—')}")
            st.markdown(f"**Context:**\n\n{doc.get('context', '—')}")
            if "annotations" in doc:
                st.markdown("**Annotations:**")
                st.json(doc["annotations"])
    else:
        st.warning(f"No full document found for source `{source_id}`.")

    #st.markdown("---")
