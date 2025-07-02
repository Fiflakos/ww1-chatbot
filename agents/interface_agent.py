# agents/interface_agent.py

import os
import streamlit as st
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data_cleaned"      # your corpus folder
FAISS_DIR = "vectorstore/faiss_index"

# 1) Load & cache retriever + generator
@st.cache_resource
def load_services():
    # embeddings for FAISS
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # load your prebuilt FAISS index (allow pickle since it's your own data)
    store = FAISS.load_local(
        FAISS_DIR,
        embed,
        allow_dangerous_deserialization=True
    )
    retriever = store.as_retriever(search_kwargs={"k": 5})

    # huggingface text2text pipeline
    gen = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device="cpu",
        max_length=256,
        do_sample=True,
        temperature=0.7,
    )

    return retriever, gen

retriever, generator = load_services()


# 2) Initialize chat history
if "history" not in st.session_state:
    # history is a list of dicts {role: "user"|"assistant", content: str}
    st.session_state.history = []

st.title("🪖 WW1 Historical Chatbot")
st.write("Ask anything across the full WW1 corpus—no uploads needed.")


# 3) Render chat messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 4) Chat input
user_input = st.chat_input("💬 What would you like to ask?")
if not user_input:
    st.stop()

# 5) Add user message to history
st.session_state.history.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)


# 6) Retrieve & build context
docs = retriever.get_relevant_documents(user_input)
if not docs:
    reply = "Sorry, I couldn’t find any letters or diaries that mention that."
else:
    # for each doc we have: d.metadata["source"] & d.page_content
    snippets = []
    for d in docs:
        src = d.metadata.get("source", "unknown.txt")
        text = d.page_content.replace("\n", " ").strip()
        snippet = text[:200] + ("…" if len(text) > 200 else "")
        snippets.append(f"**{src}**: {snippet}")

    context = "\n\n".join(snippets)

    # 7) Build a single prompt
    prompt = f"""
You are a First World War historian assistant. Use ONLY the snippets below to answer the user’s question.
If the snippets don’t cover the question, say so.

--- Snippets ---
{context}

--- Question ---
{user_input}

Please answer in 3–5 sentences and mention the source IDs you used.
""".strip()

    # 8) Call HF generator
    out = generator(prompt)
    answer = out[0].get("generated_text", "").strip()
    reply = answer

# 9) Display assistant reply
st.session_state.history.append({"role": "assistant", "content": reply})
with st.chat_message("assistant"):
    st.markdown(reply)
