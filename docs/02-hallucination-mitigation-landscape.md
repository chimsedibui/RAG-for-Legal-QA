# Survey: Existing hallucination-mitigation approaches on the market (2025-2026)

> Research document, used as the basis for the architecture direction in [01-accuracy-and-cross-validation.md](01-accuracy-and-cross-validation.md). Sources compiled via research as of 2026-08.

## 1. Why hallucination deserves special attention in the legal domain

A Stanford RegLab/HAI study (Magesh et al., *Journal of Empirical Legal Studies* 2025) tested over 200 legal questions on commercial legal-AI products:

- **Lexis+ AI**: hallucinates >17% of the time.
- **Westlaw AI-Assisted Research**: hallucinates >34% of the time.
- LexisNexis subsequently had to walk back its "100% hallucination-free" marketing claim, restricting it to just the "linked citations" portion.

According to a public tracker, as of mid-2026 there have been **over 1,590 recorded cases** of AI-"fabricated" citations appearing in court filings worldwide; in Q1/2026 alone, total fines for AI misuse in litigation reached $145,000. ABA Formal Opinion 512 (US) affirms that lawyers bear full responsibility for AI output regardless of which tool is used.

→ Conclusion: hallucination in the legal domain is **not a theoretical risk** — today's major commercial products still hallucinate at double-digit percentage rates despite heavy investment — so any improvement must be measurable (see section 5 of document 01), not just based on a "feels better" impression.

## 2. Architectural patterns

| Technique | How it works | How easy is it to bolt onto the current pipeline |
|---|---|---|
| **Corrective RAG (CRAG)** | A small model scores retrieved chunks as correct/ambiguous/incorrect; if incorrect → re-retrieve or fall back to web search; if ambiguous → filter out the irrelevant parts before feeding into context | **Easy** — just 1 scoring step + branching, no need to change the existing retriever. Reported to reduce hallucination by ~30% in practice |
| **Self-RAG** | The model generates its own "reflection tokens" deciding whether it needs to retrieve further, and self-assesses whether the answer is supported by context | **Harder** — needs a model fine-tuned/prompted specifically for this; can be simulated with a "self-critique" prompt but is less reliable than the original |
| **FLARE / DRAGIN** | Retrieves again mid-generation when the model is "unsure" about the next span (based on logprob) | **Medium** — needs an LLM API that returns logprobs, which not every OpenAI-compatible endpoint supports |
| **RAG-Fusion** | Generates multiple question variants, retrieves separately for each, merges results via Reciprocal Rank Fusion | **Very easy** — the project already has a sub-query decomposition step, just needs to change how results are merged (RRF instead of simple dedup) |
| **Self-consistency / ensemble voting** | Re-runs the same question N times (temperature > 0) or across multiple models, majority-votes; the more consistent the answers, the more trustworthy | **Technically easy, expensive in cost** — should only be enabled conditionally for important questions |
| **Citation/groundedness verification (NLI-based)** | After generating an answer, checks whether each sentence/claim is "entailed" by the cited passage, using a dedicated NLI/cross-encoder model | **Easy, highest ROI** — this is the single most worthwhile technique per this survey |
| **Generate-Verify-Correct with verbatim quote** | Forces the LLM to quote verbatim; a script checks whether the quoted text actually exists in the source chunk (substring/fuzzy match) | **Very easy** — pure Python, no model needed, high accuracy for legal text (which has precise wording, few synonyms) |
| **Confidence scoring + abstention** | Combines retrieval score + verification score + self-consistency agreement into 1 confidence value; below a threshold → refuse to answer/escalate to a human | **Easy** — just a scoring formula, no new infrastructure |
| **Human-in-the-loop** | Treats AI output as a draft; requires a lawyer to review before real use; doesn't accept using 1 AI to verify another AI in place of human review | **Not infrastructure — it's process/UX**: flag low-confidence answers to force review, similar to a medical "consult a professional" warning |

## 3. Evaluation frameworks used in practice

| Tool | Type | When to use |
|---|---|---|
| **RAGAS** | Open source, computes faithfulness/context-precision/context-recall | Run offline in CI against a fixed eval set, not real-time |
| **DeepEval** | Similar to RAGAS, pytest-assertion style | Suited to gating in a CI pipeline (fail the build if faithfulness is below a threshold) |
| **TruLens** | Tracing/feedback function for production, often paired with Langfuse | Observing quality in real time in production |
| **Vectara HHEM (2.1)** | Open-source cross-encoder specialized in scoring groundedness | Self-hostable, much faster than using an LLM-judge (reported: ~10 minutes vs ~8 hours for the same evaluation workload), benchmark accuracy ~78.9% — **the leading candidate for `HHEMVerifier` in document 01** |
| **Patronus Lynx / Galileo Luna** | Small models specialized in detecting hallucination, faster/cheaper than LLM-as-judge | If you want to use SaaS instead of self-hosting |
| **Anthropic Citations API / Gemini Grounding with Search** | Built-in grounding features from LLM providers | Only usable if switching to Claude/Gemini; with a generic OpenAI-compatible endpoint, a separate verify step still needs to be built |

**Recommended stack for this project**: run 1 fast detector (HHEM) on all/most requests + LLM-as-judge (using the existing `LLMProvider`) on a sampled set for deeper periodic review — no need to wait for SaaS integration (Galileo/Patronus) from the start.

## 4. Legal-domain specifics — grounding against statute/case-law

- **Citation Grounding metric**: the % of citations in an answer that actually exist (matching a real node in the graph/source document set) — the project already has a similar-enough structure (`chunk_map.json`, `article_index_map.json`), so adding a check that "citation `[N]` maps correctly to a real chunk + the quoted content matches verbatim" costs almost no extra infrastructure.
- **Legal citation graph / Graph-RAG**: models the citation relationships between document-document, article-article as a graph (typically using Neo4j), used both for retrieval and for validating answers. This is a bigger investment (weeks rather than days) — only worth considering if there's a need to model complex effective/amendment relationships between legal documents (very relevant to Vietnamese law, where Decrees/Circulars often amend, supplement, or replace one another).
- Given the project's current scale (chunked by Article/Clause/Point, not whole documents), a **stripped-down** version of the graph idea — a lookup table `doc_id + article → canonical text` (already available as `article_index_map.json`) — is a reasonable starting point, no need to stand up a graph DB right away.

## 5. Web-search-augmented RAG — cross-validating against external sources

- **Common pattern**: query FAISS first; if a CRAG-style scoring step or a post-generation verify step reports low confidence → fall back to web search to cross-check, or flag "needs further verification" instead of silently returning a wrong answer.
- **Common search API providers**: Tavily (regarded as the "de facto standard" for AI agents in 2025-2026, easy to integrate), Exa (semantic search), Bing Search API, SerpAPI, You.com API — differing mainly in how results are structured and cost, not much difference in quality for the cross-check use case.
- **Note specific to Vietnamese law**: web search can be useful for detecting documents that have **expired/been amended** which the internal data store (crawled once, see `pipeline/crawl_preprocess.py`) hasn't picked up yet — this is a hallucination risk specific to the legal domain (information correct at crawl time but wrong at query time), distinct from ordinary "made-up content" hallucination.

## 6. Recommendations for this project (in priority order)

1. **Verbatim quote / citation existence check** — do this now, nearly free, catches wrong/fabricated citations.
2. **RAGAS/DeepEval in CI** against a fixed eval question set — to get measurable numbers before further optimizing, avoiding "gut feel" optimization.
3. **HHEM (or equivalent) as a runtime groundedness scorer** — low cost, high effectiveness per the benchmarks.
4. **RAG-Fusion for sub-queries** — improves retrieval without needing a new interface.
5. **Conditional web search fallback** (Tavily/Exa) — only when internal confidence is low, especially useful for catching expired documents.
6. **Self-consistency/ensemble** — save for last since it's expensive, only enable for questions flagged high-stakes.

Phased implementation details: see [01-accuracy-and-cross-validation.md](01-accuracy-and-cross-validation.md) section 4.

## References

- Magesh et al., "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools", *Journal of Empirical Legal Studies* (2025) — https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413
- CRAG (Corrective RAG) — https://github.com/HuskyInSalt/CRAG , https://openreview.net/forum?id=JnWJbrnaUE
- Self-consistency / LLM fan-out patterns — https://arxiv.org/pdf/2505.09031
- Auto-GDA (NLI-based groundedness) — https://arxiv.org/pdf/2410.03461
- Vectara HHEM benchmarking — https://cleanlab.ai/blog/rag-tlm-hallucination-benchmarking/
- DeepEval vs RAGAS vs TruLens — https://particula.tech/blog/deepeval-vs-ragas-vs-trulens-rag-evaluation-stack
- Anthropic Citations API — https://claude.com/blog/introducing-citations-api
- Gemini Grounding with Search — https://ai.google.dev/gemini-api/docs/google-search
- Legal citation graphs / Citation Grounding metric — https://arxiv.org/pdf/2606.00898 , https://arxiv.org/pdf/2605.28120
- Ontology-driven Graph RAG for legal text — https://journals.sagepub.com/doi/10.3233/FAIA251598
- Comparison of search APIs for AI agents (Tavily/Exa/...) — https://brave.com/learn/best-search-api-2026/
