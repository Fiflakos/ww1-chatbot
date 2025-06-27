# agents/interface_agent.py
import streamlit as st
from transformers import pipeline
from agents.retrieval_agent import RetrievalAgent

@st.cache_resource
def load_agents():
    retr = RetrievalAgent(corpus_dir="data_cleaned")
    gen = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=0 if __import__("torch").cuda.is_available() else -1,
    )
    return retr, gen

retriever, generator = load_agents()

st.title("🪖 WW1 Historical Chatbot")
st.write("Ask anything across the full WW1 corpus—no uploads needed.")

query = st.text_input("💬 What would you like to ask?")
if query:
    hits = retriever.search(query, top_k=5)
    st.write("📌 Top passages:")
    for i,(fn,score,txt) in enumerate(hits,1):
        st.write(f"{i}. **{fn}**  (score {score:.2f})")
        st.write(txt + "…")

    prompt = "Context:\n" + "\n\n".join(f"{fn}: {txt}" for fn,_,txt in hits)
    prompt += f"\n\nQuestion: {query}"

    with st.spinner("🧠 Generating…"):
        out = generator(prompt, max_length=128)[0]["generated_text"]
    st.markdown(f"**📜 Answer:** {out}")
