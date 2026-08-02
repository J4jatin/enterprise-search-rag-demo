"""Live demo UI for the Enterprise Search RAG API.

Reuses the project's real retrieval + generation logic (rag.py):
  RETRIEVE (Sentence Transformers + cosine similarity)
  -> AUGMENT (context injection)
  -> GENERATE (Groq Llama 3.1, grounded prompt).

The Groq key is read from Streamlit secrets so the app can call the LLM.
"""
import os
import streamlit as st

# Make the Groq key available as an env var BEFORE importing rag.py,
# because rag.py creates the Groq client at import time.
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Enterprise Search — RAG Demo",
                   page_icon="🔍", layout="wide")


@st.cache_resource
def load_rag():
    from rag import search, generate_answer
    return search, generate_answer


st.title("🔍 Enterprise Search — RAG Demo")
st.caption("Semantic retrieval (Sentence Transformers) + grounded generation "
           "(Groq Llama 3.1). Answers are anchored only in retrieved documents.")

search, generate_answer = load_rag()

q = st.text_input("Ask a question",
                  "How can we reduce cloud costs?")
top_k = st.slider("Documents to retrieve (top-k)", 1, 5, 3)

if st.button("Search"):
    with st.spinner("Retrieving documents and generating a grounded answer..."):
        docs = search(q, top_k)
        try:
            answer = generate_answer(q, docs)
        except Exception as e:
            answer = ("[LLM unavailable — add GROQ_API_KEY in the app's "
                      f"Streamlit secrets to enable generation. Details: {e}]")
    st.markdown("### Answer")
    st.write(answer)
    st.markdown("### Retrieved sources")
    for d in docs:
        st.markdown(f"**{d['title']}**  ·  similarity `{d['similarity_score']}`")
        st.caption(d["content"])
