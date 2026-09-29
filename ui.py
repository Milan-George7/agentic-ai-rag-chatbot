"""Optional Streamlit UI. Run the API first, then: streamlit run ui.py"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI eBook Chatbot")
st.caption("Answers come only from the Agentic AI eBook.")
top_k = st.sidebar.slider("Chunks to retrieve (top_k)", 1, 10, 5)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("meta"):
            m = turn["meta"]
            st.caption(f"Confidence: {m['confidence']:.2f} · grounded: {m['grounded']} · {m['reason']}")
            with st.expander("Retrieved context"):
                for c in m["contexts"]:
                    st.markdown(f"**Page {c['page']} · score {c['score']}**")
                    st.text(c["text"])

if q := st.chat_input("Ask about the eBook..."):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            r = requests.post(f"{API}/chat", json={"question": q, "top_k": top_k}, timeout=90)
            r.raise_for_status()
            data = r.json()
            st.markdown(data["answer"])
            st.caption(f"Confidence: {data['confidence']:.2f} · grounded: {data['grounded']} · {data['reason']}")
            with st.expander("Retrieved context"):
                for c in data["contexts"]:
                    st.markdown(f"**Page {c['page']} · score {c['score']}**")
                    st.text(c["text"])
            st.session_state.history.append({"role": "assistant", "content": data["answer"], "meta": data})
        except Exception as e:
            st.error(f"API error: {e}")
