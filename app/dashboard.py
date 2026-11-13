"""Streamlit Dashboard for Hybrid Search RAG."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(
    page_title="Hybrid RAG — HuggingFace Docs",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Hybrid Search RAG")
st.caption("Dense + BM25 + RRF + Cross-Encoder Reranker over Transformers & PEFT documentation")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid_rerank", "hybrid", "dense", "bm25"],
        index=0,
        help="dense = BGE embeddings; bm25 = exact match; hybrid = RRF fusion; hybrid_rerank = RRF + Cross-Encoder",
    )
    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=20, value=5)
    compare_mode = st.checkbox("Compare with Dense-only", value=False)
    
    st.divider()
    st.markdown("**Health Status:**")
    try:
        health_resp = requests.get(f"{API_URL}/healthz", timeout=3)
        if health_resp.status_code == 200:
            hdata = health_resp.json()
            st.success(f"API Online (v{hdata.get('version')})")
            st.write(f"- ChromaDB: {'✓' if hdata.get('chroma_ok') else '✗'}")
            st.write(f"- BM25: {'✓' if hdata.get('bm25_loaded') else '✗'}")
        else:
            st.error("API returned error")
    except Exception:
        st.warning("API unreachable (is `make api` running?)")


# ── Query Helper ─────────────────────────────────────────────────────────────
def query_api(question: str, req_mode: str, req_k: int):
    try:
        resp = requests.post(
            f"{API_URL}/v1/ask",
            json={"question": question, "mode": req_mode, "top_k": req_k},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, str(e)


def render_response(res: dict):
    if res.get("refused"):
        st.warning(f"⚠️ **Refused:** {res.get('refusal_reason')}")
    
    st.markdown("### Answer")
    st.write(res.get("answer", ""))

    # Confidence Metrics
    conf = res.get("confidence", {})
    st.markdown("#### Confidence Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite", f"{conf.get('composite', 0.0):.2f}")
    c2.metric("Retrieval", f"{conf.get('retrieval', 0.0):.2f}")
    c3.metric("Coverage", f"{conf.get('coverage', 0.0):.2f}")
    c4.metric("Completeness", f"{conf.get('completeness', 0.0):.2f}")

    # Citations
    citations = res.get("citations", [])
    if citations:
        with st.expander(f"📚 Citations ({len(citations)})", expanded=True):
            for c in citations:
                badge = "✅ Verified" if c.get("verified") else "⚠️ Unverified"
                st.markdown(f"**[{c.get('n')}]** {badge} — `{c.get('source')}`")
                if c.get("heading"):
                    st.caption(f"Section: {c.get('heading')}")
                if c.get("verification_reason"):
                    st.caption(f"Reason: {c.get('verification_reason')}")

    # Chunks Expander
    retrieved = res.get("retrieved", [])
    with st.expander(f"📄 Retrieved Chunks ({len(retrieved)})"):
        for chunk in retrieved:
            st.markdown(f"**Rank {chunk.get('rank')}** (Score: {chunk.get('score', 0):.4f}) — `{chunk.get('chunk_id')}`")
            st.text(chunk.get("text", "")[:400] + ("..." if len(chunk.get("text", "")) > 400 else ""))
            st.divider()

    # Latencies
    lat = res.get("latency", {})
    st.caption(
        f"⏱️ Total: {lat.get('total_ms')}ms | "
        f"Retrieve: {lat.get('retrieve_ms')}ms | "
        f"Rerank: {lat.get('rerank_ms')}ms | "
        f"Generate: {lat.get('generate_ms')}ms | "
        f"Verify: {lat.get('verify_ms')}ms"
    )


# ── Main Chat Area ───────────────────────────────────────────────────────────
question = st.text_input(
    "Enter a question about HuggingFace Transformers or PEFT:",
    placeholder="e.g., How do I configure LoRA target_modules for LLaMA?",
)

if st.button("Submit Query", type="primary") and question:
    if compare_mode:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Hybrid ({mode})")
            with st.spinner("Querying hybrid pipeline..."):
                res, err = query_api(question, mode, top_k)
                if err:
                    st.error(err)
                elif res:
                    render_response(res)
        with col2:
            st.subheader("Dense-only Baseline")
            with st.spinner("Querying dense baseline..."):
                res_dense, err_dense = query_api(question, "dense", top_k)
                if err_dense:
                    st.error(err_dense)
                elif res_dense:
                    render_response(res_dense)
    else:
        with st.spinner(f"Querying using {mode}..."):
            res, err = query_api(question, mode, top_k)
            if err:
                st.error(err)
            elif res:
                render_response(res)
