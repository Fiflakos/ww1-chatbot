# interface_agent.py

import os
from dotenv import load_dotenv

import streamlit as st
import openai
import pandas as pd
import matplotlib.pyplot as plt

from retrieval_agent import RetrievalAgent

# ──────────────────────────────────────────────────────────────────────────────
# 1) Page configuration (must be first)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🪖 WW1 Historical Chatbot", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load environment and API key
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # loads .env
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("🔑 Please set OPENAI_API_KEY in your .env file")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────
mode = st.sidebar.radio("Navigate to", ["Chat", "Analytics"])

# ──────────────────────────────────────────────────────────────────────────────
# Shared retriever
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_retriever():
    return RetrievalAgent(txt_dir="data_cleaned", json_dir="data")

retriever = get_retriever()

if mode == "Chat":
    # ──────────────────────────────────────────────────────────────────────────
    # Chat page
    # ──────────────────────────────────────────────────────────────────────────
    st.title("🪖 WW1 Historical Chatbot")
    st.write("Ask anything across the full WW1 corpus—no uploads needed.")

    # helper to generate answers
    def generate_answer(query: str, top_k: int = 5):
        hits = retriever.search(query, top_k=top_k)
        hits = [(fn, sc, sn) for fn, sc, sn in hits if sc > 0]

        if not hits:
            return "No relevant passages found.", []

        context = "\n\n".join(f"{fn}: {sn}…" for fn, _, sn in hits)
        system = (
            "You are a knowledgeable WW1 historian assistant. "
            "Answer concisely, cite entries, and ground your answer in the provided context."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
        return answer, hits

    # manage chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_hits" not in st.session_state:
        st.session_state.last_hits = []

    # input form (clears on submit)
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("💬 Your question", key="input")
        submit = st.form_submit_button("Send")

    if submit and query:
        with st.spinner("🔍 Retrieving and generating answer…"):
            answer, hits = generate_answer(query)
        st.session_state.history.append(("user", query))
        st.session_state.history.append(("bot",  answer))
        st.session_state.last_hits = hits

    # render history
    for role, text in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

    # show top passages
    if st.session_state.last_hits:
        st.subheader("📌 Top passages")
        for i, (fn, score, snippet) in enumerate(st.session_state.last_hits, start=1):
            st.markdown(f"**{i}. {fn}** _(score {score:.2f})_")
            st.write(snippet + "…")

else:
    # ──────────────────────────────────────────────────────────────────────────
    # Analytics page
    # ──────────────────────────────────────────────────────────────────────────
    st.title("📊 Evaluation Analytics")
    RESULTS_FILE = "results_all_docs.csv"
    if not os.path.exists(RESULTS_FILE):
        st.error(f"Couldn’t find {RESULTS_FILE}. Run your evaluation script first.")
        st.stop()

    # 1) Load
    df = pd.read_csv(RESULTS_FILE)

    # 2) Summary statistics
    st.subheader("Summary statistics")
    st.dataframe(
        df.describe().T,  # transpose for readability
        use_container_width=True
    )

    # 3) Metric distributions
    st.subheader("Metric distributions")
    distributions = ["rougeL_f1", "emb_cosine", "faithfulness"]
    for metric in distributions:
        st.markdown(f"**{metric}**")
        fig, ax = plt.subplots()
        ax.hist(df[metric].dropna(), bins=50)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        st.pyplot(fig)
