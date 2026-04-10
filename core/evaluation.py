"""
core/evaluation.py
------------------
RAG evaluation using RAGAS and DeepEval frameworks.
RAGAS: Computes faithfulness, answer relevancy, context precision, context recall.
DeepEval: LLM-as-judge testing with pass/fail thresholds.
Both use Gemini as the judge LLM to keep costs low.
"""

from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# RAGAS Evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(eval_samples: List[Dict], gemini_llm, embedding_model) -> Optional[object]:
    """
    Run RAGAS evaluation on a list of samples.

    Each sample in eval_samples must have:
    - question (str)
    - answer (str)      -- the RAG system's generated answer
    - contexts (list)   -- list of retrieved chunk texts
    - ground_truth (str) -- the correct reference answer (for context_recall)

    Returns a RAGAS result object or None on error.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset

        # Wrap our LLM and embeddings to be RAGAS-compatible
        ragas_llm = LangchainLLMWrapper(gemini_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

        # Create HuggingFace Dataset — required by RAGAS evaluate()
        dataset = Dataset.from_list(eval_samples)

        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        return results

    except Exception as e:
        raise RuntimeError(f"RAGAS evaluation failed: {e}")


def parse_ragas_results(ragas_results) -> Dict:
    """
    Parse RAGAS results into a clean dictionary for display.
    Returns aggregate scores and per-sample DataFrame.
    """
    df = ragas_results.to_pandas()
    numeric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    aggregate = {}
    for col in numeric_cols:
        if col in df.columns:
            aggregate[col] = round(float(df[col].mean()), 4)

    return {"aggregate": aggregate, "dataframe": df}


# ---------------------------------------------------------------------------
# DeepEval Evaluation
# ---------------------------------------------------------------------------

class GroqDeepEvalModel:
    """
    A lightweight wrapper to make any LangChain LLM work as a DeepEval judge.
    DeepEval normally expects OpenAI — this bridges the gap for Groq/Gemini.
    """

    def __init__(self, langchain_llm):
        self.langchain_llm = langchain_llm

    def generate(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        return self.langchain_llm.invoke(messages).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "custom-langchain-llm"


def run_deepeval_tests(
    test_cases_data: List[Dict],
    llm,
    hallucination_threshold: float = 0.5,
    relevancy_threshold: float = 0.7,
    faithfulness_threshold: float = 0.7,
) -> List[Dict]:
    """
    Run DeepEval tests on a list of QA samples.

    Each item in test_cases_data must have:
    - question (str)
    - answer (str)
    - contexts (list of str)

    Returns a list of result dicts with scores, pass/fail, and reasons.
    """
    try:
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.models.base_model import DeepEvalBaseLLM

        # Build a proper DeepEval-compatible model wrapper
        class _DeepEvalWrapper(DeepEvalBaseLLM):
            def __init__(self, lc_llm):
                self._lc_llm = lc_llm

            def load_model(self):
                return self._lc_llm

            def generate(self, prompt: str) -> str:
                return self._lc_llm.invoke([HumanMessage(content=prompt)]).content

            async def a_generate(self, prompt: str) -> str:
                return self.generate(prompt)

            def get_model_name(self) -> str:
                return "groq-llama"

        judge = _DeepEvalWrapper(llm)

        hallucination_metric = HallucinationMetric(threshold=hallucination_threshold, model=judge, include_reason=True)
        relevancy_metric = AnswerRelevancyMetric(threshold=relevancy_threshold, model=judge, include_reason=True)
        faithfulness_metric = FaithfulnessMetric(threshold=faithfulness_threshold, model=judge, include_reason=True)

        all_results = []
        for item in test_cases_data:
            tc = LLMTestCase(
                input=item["question"],
                actual_output=item["answer"],
                retrieval_context=item["contexts"],
                context=item["contexts"],           # for HallucinationMetric
            )

            hallucination_metric.measure(tc)
            relevancy_metric.measure(tc)
            faithfulness_metric.measure(tc)

            all_results.append({
                "question": item["question"],
                "answer": item["answer"],
                "hallucination_score": round(hallucination_metric.score, 4),
                "hallucination_passed": hallucination_metric.score <= hallucination_threshold,
                "hallucination_reason": hallucination_metric.reason,
                "relevancy_score": round(relevancy_metric.score, 4),
                "relevancy_passed": relevancy_metric.score >= relevancy_threshold,
                "relevancy_reason": relevancy_metric.reason,
                "faithfulness_score": round(faithfulness_metric.score, 4),
                "faithfulness_passed": faithfulness_metric.score >= faithfulness_threshold,
                "faithfulness_reason": faithfulness_metric.reason,
                "overall_passed": (
                    hallucination_metric.score <= hallucination_threshold
                    and relevancy_metric.score >= relevancy_threshold
                    and faithfulness_metric.score >= faithfulness_threshold
                ),
            })

        return all_results

    except Exception as e:
        raise RuntimeError(f"DeepEval evaluation failed: {e}")
