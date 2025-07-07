# agents/interface_agent.py

import os
import streamlit as st
import openai
from agents.retrieval_agent import RetrievalAgent

# 1) Set your OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2) Cache the retriever so it only builds once
@st.cache_resource
def load_retriever():
    return RetrievalAgent(corpus_dir="data_cleaned")

retriever = load_retriever()

st.set_page_config(page_title="🪖 WW1 Historical Chatbot")
st.title("🪖 WW1 Historical Chatbot")
st.write("Ask anything across the full WW1 corpus—no uploads needed.")

q = st.text_input("💬 What would you like to ask?")
if not q:
    st.stop()

# 3) Retrieve top-k
hits = retriever.search(q, top_k=5)
# filter out zero-score
hits = [(fn, sc, sn) for fn, sc, sn in hits if sc > 0]

if not hits:
    st.warning("No relevant passages found.")
else:
    st.subheader("📌 Top passages")
    for i, (fn, score, snippet) in enumerate(hits, 1):
        st.markdown(f"**{i}. {fn}**  _(score {score:.2f})_")
        if snippet:
            st.write(snippet + "…")

    # 4) Build the prompt
    context = "\n\n".join(f"{fn}: {sn}" for fn, _, sn in hits)
    system = "You are a WW1 historian assistant. Answer concisely and cite which letter/diary entry you used."
    user = f"Context:\n{context}\n\nQuestion: {q}"

    # 5) Call ChatCompletion (new 1.x interface)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            temperature=0.7,
            max_tokens=256,
        )
        answer = resp.choices[0].message.content
        st.markdown(f"**📜 Answer:** {answer}")
    except Exception as e:
        st.error(f"❌ OpenAI API error: {e}")
