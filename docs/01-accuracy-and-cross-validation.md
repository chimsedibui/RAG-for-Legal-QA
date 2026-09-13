# Direction: Maximizing accuracy & multi-source cross-validation

> Architecture roadmap — not yet implemented. Based on the market survey in [02-hallucination-mitigation-landscape.md](02-hallucination-mitigation-landscape.md).

## 1. Goals

- Raise answer accuracy to the highest level achievable within the current infrastructure (FAISS + an OpenAI-compatible LLM).
- Reduce hallucination by **verifying** answers before returning them, instead of trusting a single LLM generation outright.
- Cross-validate against multiple sources (multiple retrieval passes, multiple models, and web search) when the internal source's (FAISS) confidence is low.
- Keep the interface-first architecture principle already established (see `REPORT.md`) — every new component is 1 interface in `core/interfaces.py` + 1 concrete implementation in `providers/`, without hardcoding if/else into `services/rag_pipeline.py`.

## 2. Where hallucination can occur in the current pipeline

Looking back at `services/rag_pipeline.py`, hallucination can arise in 3 places:

1. **Sub-query decomposition** — the LLM splits the question itself, which may split it wrong/incompletely → context retrieval is skewed from the start.
2. **Retrieval** — FAISS returns a chunk that isn't actually relevant (high rerank score but mismatched content), and the LLM still has to synthesize from that context.
3. **Answer generation** — the LLM may misinterpret chunk content, attach the wrong citation number `[N]`, or "invent" details not present in the context even though the citation rule is stated in the prompt.

Currently **no step re-checks** the answer after it's generated — this is the biggest gap to fill according to the survey in document 02 (the "citation/groundedness verification" technique has the highest ROI and is the easiest to bolt on).

## 3. Proposed architecture

### 3.1. New interfaces (`core/interfaces.py`)

```python
class VerificationResult(TypedDict):
    claim: str
    supported: bool          # whether it's supported by the cited source
    confidence: float        # 0..1
    reason: str               # short reason (for logging/debugging)

class Verifier(Protocol):
    def verify(self, claim: str, source_text: str) -> VerificationResult: ...

class WebSearchProvider(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Returns [{title, url, snippet}], used to cross-check when FAISS
        isn't confident enough or the document may no longer be in effect."""
        ...

class ConfidenceScorer(Protocol):
    def score(self, *, retrieval_scores: list[float], verification_results: list[VerificationResult]) -> float:
        """Combines retrieval scores + verification scores into 1 single confidence
        score, used to decide: answer directly / add web search / abstain."""
        ...
```

### 3.2. Concrete implementations (`providers/`)

| Provider | Role | Note |
|---|---|---|
| `providers/quote_verifier.py::VerbatimQuoteVerifier` | Checks whether a cited sentence matches verbatim (substring/fuzzy) against the `content` of the cited chunk | Pure Python, no model needed, **cost is close to 0** — should be done first |
| `providers/hhem_verifier.py::HHEMVerifier` | Uses Vectara HHEM (open-source, self-hostable cross-encoder) to score entailment between the answer and the source chunk | Much faster than using an LLM-judge (per survey 02); can run on CPU for a small model |
| `providers/llm_judge_verifier.py::LLMJudgeVerifier` | Uses the existing `LLMProvider`, asking once more "is this answer supported by the following passage?" | No extra infrastructure needed, but costs 1 extra LLM call per claim — should be used for sampled review, not every request |
| `providers/tavily_search.py::TavilySearchProvider` (or Exa) | Web search when cross-checking is needed | Only called when confidence is low, not called by default (to avoid adding latency/cost to every question) |
| `providers/weighted_confidence_scorer.py::WeightedConfidenceScorer` | Simple formula: `confidence = w1*avg(retrieval_score) + w2*avg(verification.confidence)` | Start with a simple linear formula, tune the weights against an eval set (see section 5) |

### 3.3. New processing step in `RAGPipeline.process()`

Add a **verification** step after `full_answer` is produced, before emitting the `answer/done` event:

```
[Step 3+4] Answer (as it currently is)
    │
    ▼
[Step 5 — NEW] Verification
    └─► Parse the [N] citations in full_answer
    └─► For each cited sentence: verify(claim, chunk_N_content)
        (run VerbatimQuoteVerifier first — cheap; only run HHEMVerifier if in doubt)
    └─► ConfidenceScorer.score(...) → overall confidence
    └─► If confidence is low:
          - Try adding a source via WebSearchProvider (if enabled)
          - Or flag the response as "needs further verification" instead of answering with certainty
          - Or abstain entirely ("Not enough grounds to answer this question accurately")
    │
    ▼
answer/done (with new fields added: "confidence", "verification": [...])
```

Proposed new SSE event to add to `core/models.py::EventStep`: `VERIFICATION = "verification"`, emitted as `{"step": "verification", "status": "done", "data": {"confidence": 0.82, "flags": [...]}}` — following the existing event pattern exactly, so the frontend can choose to display or ignore the new field without breaking the existing flow.

### 3.4. Retrieval improvements (don't need to wait on verification)

Can be done in parallel, independently of the verification step:

- **RAG-Fusion for sub-queries**: the current sub-query step already splits the question into multiple sub-queries and merges results by deduping on `chunk_id` (`_deduplicate_docs`). Upgrade to **Reciprocal Rank Fusion** (RRF) instead of simple dedup — for each chunk appearing across multiple sub-queries, accumulate a score by rank instead of just keeping the first occurrence. This is a small change inside `SemanticSearchService`/`RAGPipeline`, no new interface needed.
- **Selective self-consistency**: for questions flagged "high-stakes" (e.g. flagged by the user, or low confidence on the first pass), re-run the answer generation step N=3 times (temperature > 0), and compare the answers — if they agree, raise confidence; if they diverge, lower confidence/abstain. The cost of N× LLM calls means this should **only be enabled conditionally**, not by default for every question.

## 4. Phased implementation roadmap

| Phase | Work | Cost/risk | Why before/after |
|---|---|---|---|
| **Phase 1** | `VerbatimQuoteVerifier` — check that citation `[N]` exists in `citation_map` + that the quoted content matches the chunk verbatim/fuzzily | Nearly 0 (pure Python, no extra API calls) | Highest ROI, cheapest, catches "fabricated citation" errors — should be done right away |
| **Phase 2** | `HHEMVerifier` (self-hosted small model) + `WeightedConfidenceScorer` + abstention threshold | Medium — need to self-host 1 small model, adds 1 scoring step per request | Needs Phase 1 done first to have clean citation data as input for the verifier |
| **Phase 3** | `WebSearchProvider` (Tavily/Exa) — only called when Phase 2's confidence is low | Medium — adds external API cost, needs to handle rate limits/timeouts | Only makes sense once "when is confidence low" has been measured from Phase 2 |
| **Phase 4 (optional)** | Self-consistency/ensemble for high-stakes questions | Highest (N× LLM calls) | Should only be enabled conditionally, after a mechanism to flag "high-stakes" exists from Phase 2/3 |

## 5. Measuring effectiveness

Before optimizing, a fixed eval question set is needed (can come from the R2AI 2026 leaderboard itself or be self-built) to measure:

- **Faithfulness/groundedness** via RAGAS or DeepEval (run offline in CI, not at runtime) — see document 02's "Evaluation Frameworks" section.
- Track the number of answers that get abstained / flagged "needs verification" over time — too high an abstention rate means retrieval or the threshold needs adjusting, not that the verifier is broken.

## 6. Trade-offs to keep in mind

- Every verification/cross-validation step **adds latency** — with a pipeline that already has up to `MAX_TOOL_ITERATIONS=3` tool-call loops, verification should be considered to run in parallel (async) with streaming the answer to the client rather than blocking hard, so as not to slow down the existing streaming experience.
- Web search (Phase 3) introduces 1 data source that is **not under control** (doesn't go through the internal chunk/embed pipeline) — it needs to be made clear to the user when an answer is partly based on web search rather than solely on the processed legal-document store, similar to how `sources` is currently displayed but with a clear distinction between internal source vs web source.
