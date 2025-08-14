import os
import re
from dotenv import load_dotenv
import streamlit as st

from mistralai import Mistral
from retrieval_agent import RetrievalAgent 



st.set_page_config(page_title="🪖 WW1 Historical Chatbot", layout="wide")

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    st.error("🔑 Please set MISTRAL_API_KEY in your .env file.")
    st.stop()


@st.cache_resource
def get_mistral_client() -> Mistral:
    return Mistral(api_key=MISTRAL_API_KEY)

@st.cache_resource
def get_retriever() -> RetrievalAgent:

    return RetrievalAgent(txt_dir="data_cleaned", json_dir="data")

client = get_mistral_client()
retriever = get_retriever()

def build_system_prompt() -> str:
    return (
        "You are a WW1 historian assistant. Use ONLY the passages in 'Context' to answer. "
        "If a fact is not present in the Context, reply exactly: 'Not in the provided documents.' "
        "Do not introduce any names, dates, places, numbers, or claims that are not literally present "
        "in the Context. Write in an early-20th-century tone—formal diction, restrained emotion, "
        "and period phrasing. Avoid modern slang or invented details."
    )

def make_numbered_context(hits):
    lines = []
    for i, (fname, score, snippet) in enumerate(hits, start=1):
        sn = (snippet or "")[:500].replace("\n", " ").strip()
        lines.append(f"[{i}] ({fname}) {sn}")
    return "\n".join(lines)

INSTR_QUOTES = (
    "Include 2–3 short verbatim quotes (3–20 words each) from the Context and place them in quotes "
    "like \"…\". After each quote, add the passage tag in brackets, e.g., [1] or [3]. "
    "Finish with a final line: Sources: [n1], [n2] (list the tags you used)."
)

def make_user_prompt(question: str, hits) -> str:
    context_block = make_numbered_context(hits)
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the Context above. If the Context does not contain the answer, "
        "reply exactly: 'Not in the provided documents.' "
        + INSTR_QUOTES
    )

def call_mistral(messages, model="mistral-small-latest", max_tokens=360, temperature=0.2) -> str:
    resp = client.chat.complete(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def has_valid_citations(text: str, k: int, min_distinct: int = 2) -> bool:
    """Check that the answer contains at least `min_distinct` distinct [n] tags within 1..k."""
    tags = set(int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", text))
    valid = {n for n in tags if 1 <= n <= k}
    return len(valid) >= min_distinct

st.title("🪖 WW1 Historical Chatbot")
st.write("Grounded answers with period style. Now quoting the retrieved passages and tagging them like [n] (Mistral).")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []

with st.form("chat_form", clear_on_submit=True):
    q = st.text_input("💬 Your question")
    submitted = st.form_submit_button("Send")

if submitted and q:
    TOP_K = 5
    hits = retriever.search(q, top_k=TOP_K)

    if not hits:
        st.warning("No relevant passages found.")
    else:
        system_msg = build_system_prompt()
        user_msg   = make_user_prompt(q, hits)

        with st.spinner("✍️ Composing grounded, cited answer…"):
            answer = call_mistral(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ]
            )

            if not has_valid_citations(answer, k=len(hits), min_distinct=2):
                hard_user_msg = user_msg + (
                    "\n\nYour previous draft lacked properly tagged quotes. "
                    "Now you MUST include at least two short verbatim quotes (3–20 words) from the Context, "
                    "each immediately followed by its passage tag [n]. "
                    "End with: Sources: [n1], [n2]."
                )
                answer = call_mistral(
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": hard_user_msg},
                    ],
                    temperature=0.15,  #You can adjust with Lower temperature for more deterministic output
                )

        st.session_state.history.append(("You", q))
        st.session_state.history.append(("Bot", answer))
        st.session_state.last_hits = hits

for who, text in st.session_state.history[-12:]:
    st.markdown(f"**{who}:** {text}")

if st.session_state.last_hits:
    with st.expander(f"📌 Retrieved passages (the ONLY allowed sources) — {len(st.session_state.last_hits)} snippets"):
        for i, (fname, score, snippet) in enumerate(st.session_state.last_hits, start=1):
            st.markdown(f"**[{i}] {fname}**  _(BM25 score: {score:.2f})_")
            st.write((snippet or "").strip() + "…")
