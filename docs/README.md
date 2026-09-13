# Roadmap documentation

This folder gathers all of the project's **direction/roadmap** documentation (not yet implemented code) in one place, separate from the `REPORT.md` at the repo root (which records the architecture refactor already done). Every document here follows the module-first principle already in place (`core/` defines interfaces → `providers/`/`tools/` provide concrete implementations → `services/` consumes them through interfaces → `api/app.py` is the composition root) — the new docs don't break that principle, they only propose new interfaces/providers following the same pattern.

## Document list

| Document | Content |
|---|---|
| [01-accuracy-and-cross-validation.md](01-accuracy-and-cross-validation.md) | Architecture direction: maximizing accuracy, multi-source cross-validation (FAISS + web search), reducing hallucination — new interfaces, new processing flow, phased roadmap |
| [02-hallucination-mitigation-landscape.md](02-hallucination-mitigation-landscape.md) | A survey of existing anti-hallucination methods/tools on the market (RAG in general + legal-domain specific), used as the basis for document 01 |
| [03-cloud-deployment-aws-gcp.md](03-cloud-deployment-aws-gcp.md) | Deployment direction for AWS or GCP: comparing the two platforms layer by layer, with a concrete recommended path |
| [04-faiss-scalability-and-fault-tolerance.md](04-faiss-scalability-and-fault-tolerance.md) | FAISS scalability and fault-tolerance assessment: local measurements, correctness risks, snapshot/replica design, and acceptance criteria |

## How to read

Read in the order 02 → 01 → 03 if you want to understand "what the market is currently doing" first before getting to "what this project should do"; or jump straight to 01/03 if you just need a summary of the direction and decisions.

All proposals in these documents are **roadmap — not yet implemented**. Document 04 also records measurements and findings from the current workspace; consult it before using the older sizing assumptions in document 03. When starting to implement any part, it should be split into a separate task/issue and the corresponding document updated (marking what's done) instead of mixing roadmap docs with docs describing the actual running system.
