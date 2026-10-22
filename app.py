"""
app.py -- chat interface for the Confident Agent (Streamlit).

It wraps the existing pipeline unchanged: every question goes through
ConfidentAgent.answer(), exactly as in the benchmark runs. Nothing is released
that the pipeline would not release. A panel under each reply shows what the
pipeline did (retrieved excerpts, the critic's raw reply and verdict, the
confidence score), so a reader can see why an answer was released or withheld.

Run from the project root, with LM Studio's server running and the model loaded:
    pip install streamlit
    streamlit run app.py

Each question takes as long as in the benchmark (two to four model calls; several
minutes on a CPU laptop). The first question also builds the search index.
Set CONFIDENT_AGENT_CONFIG to use a config file other than config/config.yaml.
"""
import json
import os

import streamlit as st

from main import build_agent, load_config

CONFIG_PATH = os.environ.get("CONFIDENT_AGENT_CONFIG", "config/config.yaml")
DECISION_TEXT = {
    "ANSWER": "Released",
    "ABSTAIN": "Withheld: the agent could not verify an answer against its sources.",
}

st.set_page_config(page_title="Confident Agent", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading the SAHO corpus and building the search index (first run only)...")
def get_agent():
    return build_agent(load_config(CONFIG_PATH))


def load_samples():
    out = []
    for kind, path in (("Open-ended", "open_sample30.jsonl"), ("Multiple choice", "mcq_sample30.jsonl")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out.append((kind, r))
    return out


def run_question(question, choices, parse_mode):
    agent = get_agent()
    agent.cfg.setdefault("critic", {})["parse_mode"] = parse_mode
    resp = agent.answer(question, mcq_choices=choices)
    chunks = agent.retriever.retrieve(question).chunks   # deterministic; same excerpts the pipeline used
    excerpts = [{"source": c.chunk.source_id, "score": round(c.score, 3), "text": c.chunk.text} for c in chunks]
    return {"question": question, "choices": choices, "resp": resp.__dict__, "excerpts": excerpts, "parse_mode": parse_mode}


def render_reply(item):
    r = item["resp"]
    released = r["decision"] == "ANSWER"
    if released:
        if item["choices"] and r.get("predicted_letter"):
            letter = r["predicted_letter"]
            st.markdown(f"**{letter}. {item['choices'][letter]}**")
        else:
            st.markdown(r["answer"])
        st.success(f"{DECISION_TEXT['ANSWER']} (confidence {r['confidence']:.2f})")
    else:
        st.warning(DECISION_TEXT["ABSTAIN"])
    if r.get("citations"):
        st.caption("Sources consulted: " + ", ".join(r["citations"]))
    with st.expander("Inspect the pipeline"):
        cols = st.columns(4)
        cols[0].metric("Decision", r["decision"])
        cols[1].metric("Critic verdict", r["critic_verdict"])
        cols[2].metric("Confidence", f"{r['confidence']:.2f}")
        cols[3].metric("Retrieval", r["retrieval_verdict"])
        st.markdown(f"**Search query:** {r['retrieval_query_used']}  \n**Revisions:** {r['revision_count']}  \n"
                    f"**Critic reply read with:** {item['parse_mode']} parser"
                    + (f" (status: {r['critic_parse_status']})" if r.get("critic_parse_status") else ""))
        if r.get("ungrounded_years"):
            st.error("Year check: the draft cites year(s) not found in any excerpt: " + ", ".join(r["ungrounded_years"]))
        label = "Draft answer" if released else "Draft answer (withheld, not shown to the user)"
        st.markdown(f"**{label}:**")
        st.code(r["answer"] or "(empty)", language=None, wrap_lines=True)
        st.markdown("**Critic's raw reply:**")
        st.code(r.get("critic_raw") or "(not recorded)", language=None, wrap_lines=True)
        st.markdown("**Retrieved excerpts:**")
        for i, e in enumerate(item["excerpts"], 1):
            st.markdown(f"[{i}] `{e['source']}` (similarity {e['score']})")
            st.caption(e["text"][:900])


# ---------------- sidebar
with st.sidebar:
    st.header("Confident Agent")
    st.write("South African history questions, answered only from South African History Online "
             "excerpts and released only if an independent critic verifies them.")
    parse_mode = st.radio("Critic verdict reader", ["legacy", "robust"], index=0,
                          help="legacy = original experimental condition; robust = corrected reader (Section 6.3 of the report).")
    fmt = st.radio("Question format", ["Open-ended", "Multiple choice"])
    choices = None
    if fmt == "Multiple choice":
        st.caption("Enter the four options:")
        choices = {k: st.text_input(f"Option {k}", key=f"opt_{k}") for k in "ABCD"}
    samples = load_samples()
    if samples:
        st.divider()
        labels = ["(type your own question)"] + [f"[{k}] {r['question'][:70]}" for k, r in samples]
        pick = st.selectbox("Or try a benchmark question", labels)
        if pick != labels[0] and st.button("Ask this question"):
            kind, r = samples[labels.index(pick) - 1]
            st.session_state.pending = (r["question"], r.get("choices"))
    if st.button("Clear conversation"):
        st.session_state.history = []
    st.divider()
    st.caption("Each question makes two to four model calls through LM Studio, so a reply can take several minutes on a laptop.")

# ---------------- main chat
st.title("Confident Agent")
st.caption("A Confident Agent is not one that is always correct. It is one that knows when it should not be trusted.")
st.session_state.setdefault("history", [])

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
        if item["choices"]:
            st.caption("  ".join(f"{k}. {v}" for k, v in item["choices"].items()))
    with st.chat_message("assistant"):
        render_reply(item)

typed = st.chat_input("Ask a question about South African history")
question, q_choices = None, None
if typed:
    question = typed
    q_choices = choices if (choices and all(v.strip() for v in choices.values())) else None
    if fmt == "Multiple choice" and q_choices is None:
        st.error("Fill in all four options, or switch to open-ended.")
        question = None
elif st.session_state.get("pending"):
    question, q_choices = st.session_state.pop("pending")

if question:
    with st.chat_message("user"):
        st.markdown(question)
        if q_choices:
            st.caption("  ".join(f"{k}. {v}" for k, v in q_choices.items()))
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving evidence, drafting an answer and asking the critic..."):
                item = run_question(question, q_choices, parse_mode)
        except Exception as e:  # e.g. LM Studio not running
            st.error(f"The agent could not run: {e}. Is LM Studio's server running at the address in {CONFIG_PATH}?")
        else:
            st.session_state.history.append(item)
            render_reply(item)
