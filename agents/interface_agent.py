# agents/interface_agent.py

import os
import streamlit as st
import requests

# ─── 1) Hugging Face credentials & model ────────────────────────────────────────
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL     = "google/flan-t5-base"
HF_URL       = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# ─── 2) Build & cache your retriever once ────────────────────────────────────────
@st.cache_resource
def load_agent():
    from retrieval_agent import RetrievalAgent
    return RetrievalAgent(corpus_dir="data_cleaned")

agent = load_agent()

# ─── 3) Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="WW1 Historical Chatbot")
st.title("🪖 WW1 Historical Chatbot")
st.write("Ask anything across the full WW1 corpus—no uploads needed.")

q = st.text_input("💬 What would you like to ask?")
if q:
    # ─── 3a) Retrieve top‐k passages ─────────────────────────────────────────────
    hits = agent.search(q, top_k=5)
    st.write("📌 Top passages:")
    for i, (fn, score, snippet) in enumerate(hits, start=1):
        st.write(f"{i}. **{fn}**  (score {score:.2f})")
        st.write(snippet + "…")

    # ─── 4) Build your generation prompt ─────────────────────────────────────────
    context = "\n\n".join(f"{fn}: {snippet}" for fn, _, snippet in hits)
    prompt  = f"Context:\n{context}\n\nQuestion: {q}"

    # ─── 5) Call HF Inference API ────────────────────────────────────────────────
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature":      0.7,
            "top_p":            0.9
        },
        "options": {"wait_for_model": True}
    }

    with st.spinner("🧠 Thinking..."):
        resp = requests.post(HF_URL, headers=headers, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            # HF /models endpoint returns either a list or dict
            if isinstance(data, list) and data and "generated_text" in data[0]:
                answer = data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                answer = data["generated_text"]
            else:
                answer = str(data)
            st.markdown(f"**📜 Answer:** {answer}")
        else:
            st.error(f"❌ HF inference error {resp.status_code}: {resp.text}")

else:
    st.info("Enter a question to get started.")
