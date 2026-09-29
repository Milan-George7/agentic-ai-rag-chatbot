# Sample queries

Run these with `curl` (see README) or the Streamlit UI. Queries 1–5 are in-scope topics for an
Agentic AI eBook; query 6 is deliberately out-of-scope and must be refused.

| # | Query | Expected behaviour |
|---|-------|--------------------|
| 1 | What is Agentic AI and how does it differ from traditional or generative AI? | Answered with citations from the definition/intro pages |
| 2 | What are the core components or capabilities of an AI agent? | Answered from the eBook's description of agent capabilities |
| 3 | What benefits can enterprises expect from adopting Agentic AI? | Answered from the benefits/value sections |
| 4 | What challenges, risks or governance concerns does the eBook mention? | Answered if covered; otherwise refused |
| 5 | Which industries or use cases does the eBook discuss? | Answered from the use-case sections |
| 6 | What is the capital of France? | `answered: false` + "I couldn't find that in the eBook..." |

> I couldn't download the PDF while writing this, so the exact wording of answers is not pre-recorded here.
> Run `python -m scripts.run_samples` after ingestion to generate real outputs into `docs/sample_outputs.md`,
> and adjust queries 1–5 to match the eBook's actual table of contents if any of them get refused.
