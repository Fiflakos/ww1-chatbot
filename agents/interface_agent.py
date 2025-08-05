# agents/interface_agent.py

import os
import streamlit as st
import openai
import json
from retrieval_agent import RetrievalAgent

#from retrieval_agent import retrieve_reference
#from llm_agent import AskResponse

# 1) Set your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2) Cache the retriever so it only builds once
@st.cache_resource
def load_retriever():
    return RetrievalAgent(json_path="data/annotated2_ww1_qa.json")

retriever = load_retriever()

st.set_page_config(page_title="🪖 WW1 Historical Chatbot 🪖")
st.title("🪖 AMA1 WW1 Chatbot 🪖")
st.write("Ask me questions about the First World War")

q = st.text_input("💬 What would you like to know?")
if not q:
    st.stop()

# 3) Retrieve top-k
hits = retriever.search(q, top_k=3)

if not hits:
    st.warning("No relevant passages found.")
    st.stop()

    # Build dictionary for full context preview
source_to_context = {src: snip for src, _, snip in hits}

    # Optional: Also print to terminal logs for debugging
    #for source_id, score, snippet in hits:
        #print(f"[Unified_RAG] {source_id} | Score: {score:.4f} | Snippet: {snippet[:100]}")

    # 4) Build the prompt
context = "\n\n".join(f"{src}: {sn}" for src, _, sn in hits)
system = "You are a WW1 historian assistant. Answer concisely and cite which letter/diary entry you used."
user = f"Context:\n{context}\n\nQuestion: {q}"

cited_sources = sorted(set(src for src, _, _ in hits))
    # 5) Call ChatCompletion (new 1.x interface)
try:
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        temperature=0.7,
        max_tokens=256,
    )
    answer = resp.choices[0].message.content
    cited_str = ", ".join(cited_sources)
    answer += f"\n\n_(sources: {cited_str})_"
    st.markdown(f"**📜 Answer:** {answer}")
except Exception as e:
    st.error(f"❌ OpenAI API error: {e}")
    st.stop()

st.subheader("🔍 Top passages retrieved by Unified_RAG")
for i, (source_id, score, snippet) in enumerate(hits, 1):
    st.markdown(f"**{i}. Source ID: `{source_id}`**")
    st.markdown(f"- **Unified_RAG Score:** `{score:.4f}`")
    st.markdown(f"- **Snippet:** {snippet[:300]}…")
    #st.markdown("---")  # Divider line

    # Clickable expander to show full context
    with st.expander(f"📄 View full document: {source_id}"):
        st.write(source_to_context[source_id])




