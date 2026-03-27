"""Streamlit dashboard for SciWriter MAS — visual editorial pipeline."""

import streamlit as st
import time
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="SciWriter MAS",
    page_icon="📝",
    layout="wide",
)

# --- Header ---
st.title("📝 SciWriter MAS")
st.markdown("**Multi-agent system for scientific article generation**")
st.markdown("CrewAI agents + LangGraph orchestration + RAG knowledge base")

st.divider()

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("Settings")
    preset = st.selectbox(
        "Article format",
        options=["habr", "dzen"],
        format_func=lambda x: {"habr": "Habr (technical)", "dzen": "Dzen (popular science)"}[x],
    )
    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown("""
    1. 🔍 **Research** — web + RAG search
    2. ✍️ **Write** — draft generation
    3. ✅ **Fact-Check** — claim verification
    4. 🔄 **Review** — accept/revise loop
    5. 📝 **Edit** — style & SEO polish
    6. 📤 **Publish** — final output
    """)

# --- Main: Input ---
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input(
        "Article topic",
        placeholder="e.g., Квантовые компьютеры: прорыв 2026 года",
    )
with col2:
    generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

# --- Pipeline Execution ---
if generate_btn and topic:
    # Pipeline status display
    status_container = st.container()
    progress_bar = st.progress(0)

    steps = [
        ("🔍 Researching sources...", "research"),
        ("✍️ Writing draft...", "write"),
        ("✅ Fact-checking claims...", "fact_check"),
        ("🔄 Reviewing...", "review_gate"),
        ("📝 Editing...", "edit"),
        ("📤 Publishing...", "publish"),
    ]

    step_statuses = {}
    with status_container:
        cols = st.columns(len(steps))
        for i, (label, key) in enumerate(steps):
            with cols[i]:
                step_statuses[key] = st.empty()
                step_statuses[key].markdown(f"⏳ {label.split(' ')[0]}")

    try:
        from app.graph.workflow import create_pipeline
        from app.config import settings

        pipeline = create_pipeline()

        initial_state = {
            "topic": topic,
            "preset": preset,
            "keywords": [],
            "revision_count": 0,
            "max_revisions": settings.max_revisions,
            "draft_version": 0,
            "status": "researching",
            "log": [f"[START] Topic: {topic}, Preset: {preset}"],
        }

        # Stream pipeline execution
        final_state = None
        current_step = 0

        for event in pipeline.stream(initial_state):
            for node_name, node_output in event.items():
                # Update progress
                step_idx = next(
                    (i for i, (_, key) in enumerate(steps) if key == node_name),
                    current_step,
                )
                current_step = step_idx + 1
                progress_bar.progress(current_step / len(steps))

                # Update step status
                for i, (label, key) in enumerate(steps):
                    if i < step_idx:
                        step_statuses[key].markdown(f"✅ {label.split(' ')[0]}")
                    elif i == step_idx:
                        step_statuses[key].markdown(f"🟢 {label.split(' ')[0]}")
                    else:
                        step_statuses[key].markdown(f"⏳ {label.split(' ')[0]}")

                final_state = {**initial_state, **(final_state or {}), **node_output}

        # Mark all complete
        progress_bar.progress(1.0)
        for _, key in steps:
            step_statuses[key].markdown(f"✅ {steps[[k for _, k in steps].index(key)][0].split(' ')[0]}")

        if final_state:
            st.divider()

            # --- Results ---
            st.header("📄 Generated Article")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", final_state.get("status", "unknown"))
            with col2:
                score = final_state.get("fact_check_score", 0)
                st.metric("Fact-check score", f"{score}/10")
            with col3:
                st.metric("Revisions", final_state.get("revision_count", 0))
            with col4:
                article = final_state.get("final_article", "")
                st.metric("Characters", f"{len(article):,}")

            st.divider()

            # Article preview
            tab1, tab2, tab3, tab4 = st.tabs(["📄 Article", "🔍 Sources", "✅ Fact-Check", "📋 Log"])

            with tab1:
                st.markdown(final_state.get("final_article", "No article generated"))

            with tab2:
                st.text(final_state.get("sources", "No sources"))

            with tab3:
                st.text(final_state.get("fact_check_report", "No fact-check report"))

            with tab4:
                for entry in final_state.get("log", []):
                    st.text(entry)

            # Download button
            if final_state.get("final_article"):
                st.download_button(
                    "📥 Download article (.md)",
                    data=final_state["final_article"],
                    file_name=f"{topic[:50].replace(' ', '_')}.md",
                    mime="text/markdown",
                )

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)

elif generate_btn and not topic:
    st.warning("Please enter an article topic.")

# --- Footer ---
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "SciWriter MAS — CrewAI + LangGraph + Qdrant + Ollama"
    "</div>",
    unsafe_allow_html=True,
)
