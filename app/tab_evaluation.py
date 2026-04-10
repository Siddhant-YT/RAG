"""
app/tab_evaluation.py
----------------------
Tab 7: RAG Evaluation (RAGAS + DeepEval)
Allows the user to:
- Manually add QA samples (question + ground truth)
- Or use samples collected automatically from the Chatbot tab
- Run RAGAS evaluation (faithfulness, answer relevancy, context precision, context recall)
- Run DeepEval tests with configurable thresholds
- See per-sample scores and aggregate metrics
- Understand what each metric means
"""

import streamlit as st
import pandas as pd

from core.models import load_gemini_llm, load_groq_llm, load_embedding_model
from core.evaluation import run_ragas_evaluation, parse_ragas_results, run_deepeval_tests
from core.rag_engine import simple_rag


def render():
    """Render the Evaluation tab."""
    st.header("RAG Evaluation")
    st.write(
        "Evaluate your RAG system's quality using RAGAS framework. "
        "It uses an LLM as judge. Gemini is used here for better evaluation quality."
    )

    if not st.session_state.get("kb_ready"):
        st.warning("Please build the knowledge base first in the 'Knowledge Base' tab.")
        return

    vectorstore = st.session_state["vectorstore"]
    groq_llm = load_groq_llm()
    embedding_model = load_embedding_model()

    try:
        gemini_llm = load_gemini_llm()
        gemini_available = True
    except Exception:
        gemini_available = False
        st.warning("GOOGLE_API_KEY not found. RAGAS evaluation requires Gemini as judge. Add it to .env")

    # ---- Metric Explanations ----
    with st.expander("What do these metrics measure?", expanded=False):
        # col1, col2 = st.columns(2)
        # with col1:
            st.markdown("**RAGAS Metrics**")
            st.write("**Faithfulness (0-1):** Are all claims in the answer supported by the retrieved context? "
                     "Score of 1 = fully grounded, 0 = hallucinated.")
            st.write("**Answer Relevancy (0-1):** Does the answer actually address the question? "
                     "Penalizes off-topic or incomplete answers.")
            st.write("**Context Precision (0-1):** What fraction of retrieved chunks are relevant to the question? "
                     "High score = retriever is precise.")
            st.write("**Context Recall (0-1):** Was all the information needed to answer the question retrieved? "
                     "Requires a ground truth answer.")
        # with col2:
        #     st.markdown("**DeepEval Metrics**")
        #     st.write("**Hallucination:** Detects information in the answer that is not present in the context. "
        #              "Lower is better. Threshold: answer must be below this score to pass.")
        #     st.write("**Answer Relevancy:** Checks if the answer is on-topic for the question. "
        #              "Higher is better.")
        #     st.write("**Faithfulness:** Cross-checks answer claims against retrieved context. "
        #              "Higher is better.")

    st.divider()

    # ---- Sample Management ----
    st.subheader("Evaluation Samples")

    # Samples collected from chatbot
    auto_samples = st.session_state.get("eval_samples", [])
    st.write(f"Auto-collected from chatbot: **{len(auto_samples)}** sample(s).")

    # Manual sample entry
    with st.expander("Add a manual sample", expanded=False):
        m_question = st.text_input("Question", key="eval_q")
        m_ground_truth = st.text_input("Ground truth answer (required for context recall)", key="eval_gt")
        if st.button("Add Sample") and m_question.strip():
            # Run RAG to get the answer and context
            with st.spinner("Running RAG to get answer for this sample..."):
                result = simple_rag(m_question.strip(), vectorstore, groq_llm, k=3)
            st.session_state["eval_samples"].append({
                "question": m_question.strip(),
                "answer": result["answer"],
                "contexts": [d.page_content for d in result["docs"]],
                "ground_truth": m_ground_truth.strip(),
            })
            st.success("Sample added.")
            st.rerun()

    # Fill in ground truth for auto-collected samples
    if auto_samples:
        with st.expander("Add ground truth to auto-collected samples", expanded=False):
            updated = False
            for i, sample in enumerate(auto_samples):
                gt = st.text_input(
                    f"Ground truth for: {sample['question'][:60]}...",
                    value=sample.get("ground_truth", ""),
                    key=f"gt_{i}",
                )
                auto_samples[i]["ground_truth"] = gt
                if gt:
                    updated = True
            if updated:
                st.session_state["eval_samples"] = auto_samples

    if st.button("Clear all samples", type="secondary"):
        st.session_state["eval_samples"] = []
        st.rerun()

    st.divider()

    # ---- RAGAS Evaluation ----
    st.subheader("RAGAS Evaluation")

    eval_samples = st.session_state.get("eval_samples", [])

    if not eval_samples:
        st.info("No evaluation samples yet. Use the chatbot or add samples manually above.")
    else:
        st.write(f"Ready to evaluate **{len(eval_samples)}** sample(s).")

        # Preview table
        preview_data = []
        for s in eval_samples:
            preview_data.append({
                "Question": s["question"][:70] + "..." if len(s["question"]) > 70 else s["question"],
                "Has Ground Truth": "Yes" if s.get("ground_truth") else "No",
                "Contexts": len(s.get("contexts", [])),
            })
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

        if st.button("Run RAGAS Evaluation", type="primary", disabled=not gemini_available):
            with st.spinner("RAGAS is evaluating... each sample requires multiple LLM calls. Please wait."):
                try:
                    ragas_results = run_ragas_evaluation(eval_samples, gemini_llm, embedding_model)
                    parsed = parse_ragas_results(ragas_results)

                    st.subheader("Aggregate Scores")
                    agg = parsed["aggregate"]

                    cols = st.columns(len(agg))
                    metric_labels = {
                        "faithfulness": "Faithfulness",
                        "answer_relevancy": "Answer Relevancy",
                        "context_precision": "Context Precision",
                        "context_recall": "Context Recall",
                    }
                    for col, (key, value) in zip(cols, agg.items()):
                        label = metric_labels.get(key, key)
                        # Color code: green >= 0.7, yellow >= 0.5, red < 0.5
                        delta = "Good" if value >= 0.7 else ("Moderate" if value >= 0.5 else "Low")
                        col.metric(label, f"{value:.3f}", delta)

                    st.subheader("Per-Sample Scores")
                    df = parsed["dataframe"]
                    display_cols = [c for c in ["question", "faithfulness", "answer_relevancy",
                                                "context_precision", "context_recall"] if c in df.columns]
                    st.dataframe(df[display_cols], use_container_width=True)

                except Exception as e:
                    st.error(f"RAGAS evaluation failed: {e}")

    st.divider()

    # # ---- DeepEval Evaluation ----
    # st.subheader("DeepEval Evaluation")
    # st.write(
    #     "DeepEval uses a pass/fail threshold system. "
    #     "Set thresholds below and run tests to see which samples pass."
    # )

    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     hallucination_threshold = st.slider(
    #         "Hallucination threshold (max allowed)", 0.0, 1.0, 0.5, 0.05,
    #         help="Fail if hallucination score exceeds this value. Lower = stricter."
    #     )
    # with col2:
    #     relevancy_threshold = st.slider(
    #         "Relevancy threshold (min required)", 0.0, 1.0, 0.7, 0.05,
    #         help="Fail if relevancy score is below this value."
    #     )
    # with col3:
    #     faithfulness_threshold = st.slider(
    #         "Faithfulness threshold (min required)", 0.0, 1.0, 0.7, 0.05,
    #         help="Fail if faithfulness score is below this value."
    #     )

    # if eval_samples:
    #     if st.button("Run DeepEval Tests", type="primary"):
    #         with st.spinner("DeepEval is running tests... this may take a moment per sample."):
    #             try:
    #                 deepeval_results = run_deepeval_tests(
    #                     test_cases_data=eval_samples,
    #                     llm=groq_llm,
    #                     hallucination_threshold=hallucination_threshold,
    #                     relevancy_threshold=relevancy_threshold,
    #                     faithfulness_threshold=faithfulness_threshold,
    #                 )

    #                 # Summary
    #                 total = len(deepeval_results)
    #                 passed = sum(1 for r in deepeval_results if r["overall_passed"])

    #                 col1, col2, col3 = st.columns(3)
    #                 col1.metric("Total Tests", total)
    #                 col2.metric("Passed", passed)
    #                 col3.metric("Failed", total - passed)

    #                 # Per-sample table
    #                 st.subheader("Test Results")
    #                 table_data = []
    #                 for r in deepeval_results:
    #                     table_data.append({
    #                         "Question": r["question"][:60] + "...",
    #                         "Hallucination": f"{r['hallucination_score']:.3f} ({'PASS' if r['hallucination_passed'] else 'FAIL'})",
    #                         "Relevancy": f"{r['relevancy_score']:.3f} ({'PASS' if r['relevancy_passed'] else 'FAIL'})",
    #                         "Faithfulness": f"{r['faithfulness_score']:.3f} ({'PASS' if r['faithfulness_passed'] else 'FAIL'})",
    #                         "Overall": "PASS" if r["overall_passed"] else "FAIL",
    #                     })
    #                 st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    #                 # Reasoning detail per sample
    #                 for r in deepeval_results:
    #                     with st.expander(f"Details: {r['question'][:60]}..."):
    #                         st.write(f"**Hallucination Reason:** {r.get('hallucination_reason', 'N/A')}")
    #                         st.write(f"**Relevancy Reason:** {r.get('relevancy_reason', 'N/A')}")
    #                         st.write(f"**Faithfulness Reason:** {r.get('faithfulness_reason', 'N/A')}")

    #             except Exception as e:
    #                 st.error(f"DeepEval evaluation failed: {e}")
    # else:
    #     st.info("Add evaluation samples above to run DeepEval tests.")
