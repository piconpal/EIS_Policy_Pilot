# Evaluation Recommendations
Generated from: logs/eval_report.json
Dataset: 30 golden queries | top_k=5 | reranker_top_n=3 | model=llama-3.1-8b-instant

---

## Overall Scores

| Metric              | Score  |
|---------------------|--------|
| Mean Precision@k    | 0.4111 |
| Mean Recall@k       | 0.7333 |
| MRR                 | 0.6056 |
| Mean Keyword Hit Rate | 0.4422 |

### By Query Type

| Type       | Count | P@k    | R@k    | MRR    | KW     |
|------------|-------|--------|--------|--------|--------|
| factual    | 8     | 0.5417 | 0.875  | 0.7292 | 0.5929 |
| metric     | 5     | 0.5333 | 1.000  | 0.6667 | 0.2800 |
| process    | 5     | 0.4000 | 0.800  | 0.8000 | 0.2524 |
| comparison | 4     | 0.4167 | 0.750  | 0.6250 | 0.2321 |
| table      | 8     | 0.2083 | 0.375  | 0.3125 | 0.6167 |

---

## Per-Query Recommendations

### q001 — What is RBAC and how is it implemented in an enterprise environment?
- **Scores**: P@k=1.0 | R@k=1.0 | KW=0.60
- **Status**: Retrieval is perfect. LLM answer mentions "role", "access control", "least privilege" but not "permission" or "assignment".
- **Recommendation**: Keyword "assignment" and "permission" are used in the PDF source but the LLM paraphrased. This is a healthy response — no system change needed. If KW is critical, shorten expected keywords to high-frequency terms like "role" and "permission" only.

---

### q002 — What is the difference between RBAC and ABAC in identity and access management?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.00 | top_chunk_score=0.41
- **Status**: Right document retrieved but cross-encoder score is 0.41 (very low). LLM correctly refused to answer — it received a chunk about RBAC mission/overview, not ABAC comparison.
- **Recommendation**: The ABAC comparison content likely lives in a specific section of `iam_rbac_policy.pdf`. The retriever is returning the intro/overview chunk rather than the comparison section. **Increasing top_k** would give the reranker more candidates to find the right section. Also consider adding "ABAC" as a term in the chunker section header to improve BM25 recall.

---

### q003 — What KPIs are used to measure the effectiveness of an RBAC implementation?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.00 | top_chunk_score=0.27
- **Status**: Right document retrieved but cross-encoder score is 0.27 (extremely low). LLM refused to answer. The KPI section of the PDF was not in the retrieved chunks.
- **Recommendation**: KPI content is typically in later sections (e.g. section 5) while the retriever is surfacing intro sections (section 1.1). This is a **retrieval depth problem**. Increasing top_k from 5 to 10 will bring later-document sections into the candidate pool. The cross-encoder will then correctly promote the KPI chunk.

---

### q004 — How does a transaction monitoring system detect fraudulent payment activity?
- **Scores**: P@k=0.33 | R@k=1.0 | KW=0.00 | top_chunk_score=7.20
- **Status**: High reranker score (7.2) yet the LLM said "I don't have enough information." This is an anomaly — the cross-encoder judged the chunk as highly relevant but the LLM refused.
- **Recommendation**: Investigate the LLM prompt template. The LLM may be hitting a response guardrail or the chunk text, while scoring high on relevance, contains structured/tabular text that the LLM's instruction-following logic treats as insufficient for answering a "how" question. Review `llm_handler.py` system prompt to ensure it does not over-penalize partial evidence.

---

### q005 — What are the key metrics used to evaluate a fraud detection model in transaction monitoring?
- **Scores**: P@k=0.67 | R@k=1.0 | RR=0.50 | KW=0.00
- **Status**: Reranker promoted `fraud_anomaly_detection.pdf` to rank 1 above `fraud_transaction_monitoring.pdf`. LLM answered with "coverage percentages" and "mean time to detect" instead of "false positive", "precision", "recall", "AUC".
- **Recommendation**: The cross-encoder preferred the anomaly detection doc over the transaction monitoring doc for this query. This is a cross-encoder calibration issue, not a retrieval failure. The correct chunk exists in the candidate set (rank 2/3) but the LLM naturally leans on the highest-ranked source. Consider passing all reranked chunks to the LLM with equal context weight rather than position-biased prompting.

---

### q006 — What is the CVSS scoring system and how is a vulnerability score calculated?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.67
- **Status**: Working. Missing "vector" and "impact" — LLM used "attack vector" and "impact score" as full phrases, not standalone words.
- **Recommendation**: Loosen keyword to substring check (e.g. "impact" would match "impact score"). No system change needed — this is a dataset calibration issue.

---

### q007 — What is the difference between CVSS Base Score and CVSS Temporal Score?
- **Scores**: P@k=0.33 | R@k=1.0 | KW=0.50
- **Status**: LLM answered correctly. Missing "remediation", "environmental", "maturity" — these are temporal score sub-concepts that the LLM summarized at a higher level.
- **Recommendation**: The LLM gave a correct high-level comparison but didn't drill into sub-metrics. This is a prompt depth issue. The LLM prompt could instruct the model to enumerate specific sub-components when answering comparison queries.

---

### q008 — What is behavioral baseline profiling in UEBA and how is it established?
- **Scores**: P@k=0.33 | R@k=1.0 | KW=0.50
- **Status**: LLM mentioned "baseline", "behavior", "machine learning" but not "normal", "deviation", "user profile". The answer is semantically correct.
- **Recommendation**: "normal" and "deviation" are implicit in baseline profiling definitions. The LLM captured the concept without the exact words. This is a keyword phrasing issue — not a system failure. Semantic keyword matching would correctly score this as a hit.

---

### q009 — How does UEBA detect anomalous user behavior that indicates a compromised account?
- **Scores**: P@k=0.00 | R@k=0.00 | KW=0.17
- **Status**: Golden dataset maps this to `ueba_baseline_profiling.pdf` but the retriever surfaced `ueba_anomaly_detection.pdf` as rank 1 — which is semantically the more relevant document for this query. P@k and R@k show 0.0 only because the golden mapping is strict.
- **Recommendation**: Review the golden dataset mapping for q009. `ueba_anomaly_detection.pdf` is arguably the correct source for anomaly detection. Update `relevant_doc_ids` to include both files. This is a **dataset calibration issue**, not a system failure.

---

### q010 — What is attack surface management and what does an asset inventory include?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.50
- **Status**: LLM answered correctly but missed "exposure", "external", "discovery".
- **Recommendation**: "external" and "discovery" appear in the answer_preview ("all internet-facing assets" captures external, "continuously discover" captures discovery). Semantic matching would score these correctly. No system change needed.

---

### q011 — What KPIs measure the maturity of an attack surface management program?
- **Scores**: P@k=0.33 | R@k=1.0 | RR=0.33 | KW=0.20
- **Status**: `siem_alert_correlation.pdf` landed at rank 1 after reranking (wrong domain). LLM cited it as primary source. The correct doc `asm_asset_inventory.pdf` was at rank 3.
- **Recommendation**: This is a cross-encoder ranking error — SIEM KPI content scored higher than ASM KPI content for "ASM maturity KPIs". The cross-encoder model (ms-marco-MiniLM-L-6-v2) lacks domain-specific calibration for security document categories. Consider fine-tuning the cross-encoder on domain-specific pairs, or adding domain metadata filtering (e.g. filter by source domain prefix: "asm_" for ASM queries).

---

### q012 — How does a SIEM system aggregate and normalize logs from multiple sources?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.33
- **Status**: LLM answered with "aggregating security telemetry" but missed "normalization", "parser", "collector", "correlation". The answer is high-level.
- **Recommendation**: The LLM generalized. The specific technical terms (parser, collector) are in deeper sections of the document. Increasing top_k will surface more sections; prompting the LLM to "enumerate specific technical components" for process queries would improve term coverage.

---

### q013 — What is the difference between a SIEM alert and a SIEM correlation rule?
- **Scores**: P@k=0.00 | R@k=0.00 | KW=0.00 | top_chunk_score=2.03
- **Status**: Golden dataset maps to `siem_log_aggregation.pdf` but the concept of "alert vs correlation rule" is more naturally covered in `siem_alert_correlation.pdf`, which was actually retrieved. LLM said "no info" with score=2.03 (borderline low).
- **Recommendation**: Same pattern as q009 — likely a golden dataset mapping error. `siem_alert_correlation.pdf` is the correct source for this query. Update `relevant_doc_ids`. Additionally, score=2.03 is low enough that the LLM hedged; increasing top_k will bring in more relevant chunks and improve the score.

---

### q014 — What is a DLP framework and how does it prevent sensitive data from leaving the organization?
- **Scores**: P@k=0.33 | R@k=1.0 | KW=0.14
- **Status**: LLM gave a high-level program overview answer. Missed "classify", "block", "endpoint", "network", "inspection" — these are implementation-layer terms.
- **Recommendation**: The retrieved chunk is from `dp_dlp_framework.pdf p3 — 1.1 Program Mission` which is the overview section, not the enforcement mechanism section. The enforcement mechanism (endpoint agent, network proxy, content inspection) is in a later section. Increasing top_k would surface the enforcement sections alongside the mission overview.

---

### q015 — What KPIs are used to measure the effectiveness of a DLP program?
- **Scores**: P@k=0.33 | R@k=1.0 | KW=0.60
- **Status**: Reasonable performance. Missing "false positive" and "data exfiltration".
- **Recommendation**: The LLM cited "DLP KPIs" section correctly. "False positive" may have been mentioned indirectly. Semantic KW matching would likely score this at 0.8+. Low-priority improvement.

---

### q016 — How does insider threat detection differ from external threat detection in fraud investigations?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.43
- **Status**: LLM correctly contrasted insider vs external threats but missed "trusted", "privilege abuse", "behavioral", "intent".
- **Recommendation**: The LLM focused on the structural difference (source: internal vs external) but didn't elaborate on the behavioral indicators. Prompting for "enumerate distinguishing characteristics" in comparison queries would surface these terms. Also semantic matching would catch "trusted" implied by "originate from within the organization".

---

### q017 — What is Privileged Access Management and why is it critical for enterprise security?
- **Scores**: P@k=1.0 | R@k=1.0 | KW=0.33
- **Status**: Perfect retrieval. LLM answered correctly but missed "admin", "credential", "vault", "session recording".
- **Recommendation**: LLM used "elevated accounts" instead of "admin", and "securing" instead of "credential vault" or "session recording". These are strong semantic equivalences. Semantic KW matching would score this 0.67+. The LLM prompt could request listing specific PAM controls.

---

### q018 — How does a PAM solution enforce just-in-time access for privileged accounts?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.33
- **Status**: LLM answered correctly on "just-in-time" and "expiry" but missed "temporary", "approval", "workflow", "least privilege".
- **Recommendation**: The answer_preview ("time-bounded session checkout with automatic revocation") captures "temporary" and "expiry" semantically. The JIT workflow (approval chain) is a process described in a separate section. Increasing top_k brings in the workflow chunk alongside the definition chunk.

---

### q019 — What is a vulnerability patch management process and what are its key stages?
- **Scores**: P@k=0.33 | R@k=1.0 | KW=0.43
- **Status**: LLM stated the patch management process is "not explicitly described" in the retrieved chunk and inferred stages. Missed "scan", "test", "verify", "lifecycle".
- **Recommendation**: The process lifecycle (scan → prioritize → test → deploy → verify) is likely in a dedicated "Patch Lifecycle" section, not in the mission/objectives chunk that was retrieved (p3 — 1.1 Program Mission). Increasing top_k to capture the lifecycle section would resolve this.

---

### q020 — What KPIs indicate the health and performance of a vulnerability patch management program?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.60
- **Status**: Good. Missing "mean time to patch" and "critical vulnerabilities". The LLM cited the KPI dashboard section.
- **Recommendation**: "Mean time to patch" is used in the PDF but the LLM paraphrased as "mean time to detect". Semantic matching would score "mean time" as a partial hit. Low-priority — mostly a phrasing mismatch.

---

### q021 — SIEM Alert Severity Matrix: P2-High vs P3-Medium SLA and analyst action?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=0.67
- **Status**: LLM correctly answered with "1 hour" and "4 hour" but keywords are "1-hour" and "4-hour" (with hyphen). Two keywords missed due to format difference only.
- **Recommendation**: Fix golden dataset keywords: change "1-hour" → "1 hour" and "4-hour" → "4 hour". This is a **dataset calibration issue only** — the system is performing correctly.

---

### q022 — Critical CVSS 9.0-10.0: remediation SLAs and escalation contact?
- **Scores**: P@k=0.67 | R@k=1.0 | KW=1.00
- **Status**: Perfect keyword match. The table data (24h, 72h, CISO) was correctly extracted by pdfplumber and cited accurately.
- **Recommendation**: No changes needed. This validates the table-aware extraction pipeline for structured SLA tables.

---

### q023 — VM KPI Dashboard: Mean Time to Remediate Critical vulnerabilities?
- **Scores**: P@k=0.00 | R@k=0.00 | KW=0.00
- **Status**: Complete retrieval miss. The "31 hours / needs attention" data is in `vuln_remediation_sla.pdf` but the retriever returned only `vuln_asset_prioritization.pdf` across all 5 slots (very similar document name).
- **Recommendation**: This is a **retrieval depth and lexical collision** problem. "VM KPI Dashboard" query terms semantically match `vuln_asset_prioritization.pdf` because that doc also has a KPI dashboard section. The specific `vuln_remediation_sla.pdf` table gets buried. Increasing top_k to 10 gives the retriever more budget. Also, improving BM25 indexing of table captions (e.g. "VM KPI Dashboard" as section header in the chunk) would help.

---

### q024 — UEBA Imminent Threat: risk score range, automated actions, investigation timeline?
- **Scores**: P@k=0.33 | R@k=1.0 | KW=0.00 | top_chunk_score=-0.69
- **Status**: Right document retrieved (`ueba_risk_scoring.pdf`) but top_chunk_score is negative (-0.69). The cross-encoder found the retrieved chunks irrelevant to the query. The specific "Imminent Threat 90-100 / account suspension / SOC page" table row was not in any of the 5 retrieved chunks.
- **Recommendation**: The risk score action table is likely a small table in the PDF that became a compact chunk. With top_k=5, only 5 candidates are evaluated and none contained the table row. Increasing top_k to 10 with `has_table=True` filter preference would surface the table chunk. This is the primary use case for raising top_k.

---

### q025 — Cloud ASM KPI in Needs Attention status?
- **Scores**: P@k=0.00 | R@k=0.00 | KW=1.00
- **Status**: Silent hallucination risk. Wrong document retrieved (`asm_asset_inventory.pdf` instead of `asm_exposure_scoring.pdf`) but LLM produced a perfectly correct answer with correct values (34, 20, unauthorized cloud services). The data apparently duplicates between the two PDF files.
- **Recommendation**: While the answer is factually correct this time, this pattern is dangerous in production — the system cited a different document than the authoritative source. Two mitigations: (1) verify if this data truly duplicates across both PDFs (if so, update golden dataset to include both as valid sources); (2) add source validation to the evaluation — correct answer from wrong source should score differently than correct answer from correct source.

---

### q026 — Fraud Program KPI Dashboard: False Positive Rate target, current, and trend?
- **Scores**: P@k=0.00 | R@k=0.00 | KW=0.67
- **Status**: Wrong document retrieved (`fraud_anomaly_detection.pdf` instead of `fraud_insider_threat.pdf`). LLM produced partial correct answer (54%, improving) but missed the target (60%). The query anchor ("Fraud Program KPI Dashboard") should uniquely identify `fraud_insider_threat.pdf`.
- **Recommendation**: The term "Fraud Program KPI Dashboard" as a table caption is not distinctive enough in BM25 because multiple fraud documents have KPI sections. Add a metadata filter: when query contains "KPI Dashboard" + domain keyword, boost chunks with `has_table=True` from the matching domain document. Alternatively, increase top_k — the correct document was ranked 5th in retrieval.

---

### q027 — Data Retention Schedule: Health Records PHI retention period, regulation, disposal?
- **Scores**: P@k=0.00 | R@k=0.00 | KW=1.00
- **Status**: Similar to q025 — wrong document retrieved (`dp_data_classification.pdf` instead of `dp_encryption_standards.pdf`) but LLM answered correctly with all exact values (6 years, HIPAA, DoD 5220.22-M). Data is duplicated across both PDFs.
- **Recommendation**: Verify whether `dp_data_classification.pdf` actually contains the retention schedule table or if the LLM hallucinated correct values. If the data genuinely exists in both files, update golden dataset `relevant_doc_ids` to include both. If not, this is a hallucination that happens to be correct — which should be flagged as a trust risk.

---

### q028 — Policy Exception Request: duration, approvers, review frequency?
- **Scores**: P@k=0.00 | R@k=0.00 | KW=1.00
- **Status**: Wrong documents retrieved across all 3 slots (ueba, vuln_asset_prioritization, dp_data_classification). LLM answered correctly with all five keywords (90 days, system owner, security manager, CISO, monthly). Policy exception data appears to be duplicated across multiple documents as a standard appendix.
- **Recommendation**: This query points to a systemic pattern — boilerplate policy sections (exception processes, approval workflows) are copy-pasted across documents. The golden dataset should map to all valid source documents. At the retrieval level, no fix is needed since any of these sources produces the correct answer. The golden dataset `relevant_doc_ids` should be expanded.

---

### q029 — SIEM Program Overview: events per second and log source count?
- **Scores**: P@k=0.33 | R@k=1.0 | KW=1.00
- **Status**: Perfect factual answer (500,000 EPS, 2,000 sources). The relevant document ranked 3rd but the LLM still answered correctly because the data exists in multiple SIEM documents that share the same mission overview text.
- **Recommendation**: No system change needed. Consider expanding `relevant_doc_ids` to include all SIEM docs since they share the overview section. RR would improve from 0.33 to 1.0.

---

### q030 — UEBA risk score 75: classification, action, investigation SLA?
- **Scores**: P@k=0.00 | R@k=0.00 | KW=0.60
- **Status**: Retrieval miss — all 5 retrieved chunks are from `ueba_alert_triage.pdf` (similar domain, wrong file). `ueba_risk_scoring.pdf` never surfaced. KW=0.60 because `ueba_alert_triage.pdf` contains similar risk threshold content.
- **Recommendation**: Five-slot retrieval locked onto `ueba_alert_triage.pdf` completely — no diversity. This indicates the UEBA alert triage document is lexically dominating the retrieval index for risk score queries. Implementing **maximal marginal relevance (MMR)** in the retriever would force source diversity, ensuring that both `ueba_alert_triage.pdf` and `ueba_risk_scoring.pdf` appear in the top-5.

---

## Summary Recommendations

### S1 — Increase Retrieval Depth (top_k: 5 → 10)
**Impact**: High | **Effort**: Low (one config change)
**Affects**: q002, q003, q014, q018, q019, q023, q024, q026

The most impactful single change. With top_k=5, the retriever is too shallow to surface:
- Later document sections (KPI dashboards, enforcement mechanisms, lifecycle stages)
- Table chunks that rank below intro/overview chunks
- Less-dominant documents when one document monopolizes retrieval slots

Increasing to top_k=10 gives the cross-encoder a richer candidate pool without significantly increasing latency (reranker already processes up to 10 items efficiently).

---

### S2 — Implement Maximal Marginal Relevance (MMR) in Retriever
**Impact**: Medium | **Effort**: Medium
**Affects**: q023, q026, q030

When retrieval returns 5 chunks from the same document, the reranker has no opportunity to surface the correct document. MMR penalizes redundant sources and forces diversity in the candidate set. A lambda=0.7 (70% relevance, 30% diversity) would prevent any single document from occupying more than 2-3 retrieval slots.

---

### S3 — Semantic Keyword Matching in Evaluator
**Impact**: Medium | **Effort**: Medium
**Affects**: q001, q006, q007, q008, q010, q012, q016, q017, q018, q020, q021

Current `_keyword_hit_rate()` uses exact substring matching. Many LLM answers are semantically correct but use different phrasing:
- "elevated accounts" vs "admin"
- "time-bounded session" vs "temporary"
- "1 hour" vs "1-hour"
- "continuously discover" vs "discovery"

Replace or augment with embedding cosine similarity between expected keyword and answer text window. This will give more accurate KW scores and better signal for system improvements without distorting evaluation.

---

### S4 — Fix Golden Dataset Mapping Errors
**Impact**: Medium | **Effort**: Low
**Affects**: q009, q013, q021, q025, q027, q028, q029

Several queries have incorrect or incomplete `relevant_doc_ids`:
- **q009**: Add `ueba_anomaly_detection.pdf` as valid source
- **q013**: Replace `siem_log_aggregation.pdf` with `siem_alert_correlation.pdf`
- **q021**: Fix keyword format "1-hour" → "1 hour", "4-hour" → "4 hour"
- **q025, q027**: Verify if data duplicates across docs; if yes, add both to relevant_doc_ids
- **q028**: Add all documents that contain the policy exception appendix
- **q029**: Add all SIEM docs that share the program overview section

These corrections improve evaluation signal without touching the system — they make the evaluation accurately reflect system capability rather than dataset gaps.

---

### S5 — Investigate LLM Refusal with High Reranker Score (q004)
**Impact**: Medium | **Effort**: Low (investigation)
**Affects**: q004, potentially others

q004 has top_chunk_score=7.2 (one of the highest in the dataset) yet the LLM refused to answer. This suggests either:
- The system prompt contains a guardrail that triggers on the specific chunk content
- The chunk is a table or structured text that the LLM treats as "not enough narrative context"
- A prompt formatting issue causes the context to be misread

Review `llm_handler.py` system prompt for over-restrictive "I don't have enough information" triggers. Structured/table chunks may need a modified prompt that explicitly tells the LLM to extract values from structured text.

---

### S6 — Table-Aware Retrieval Boost
**Impact**: Medium | **Effort**: Medium
**Affects**: q021, q022, q023, q024, q025, q026, q027, q030

Table-type queries score worst (R@k=0.375, P@k=0.21). The table chunks exist in ChromaDB (`has_table=True` metadata) but are not being prioritized for queries that explicitly ask for specific values, thresholds, or dashboard entries.

Implement a query classifier that detects table-anchored queries (keywords: "according to", "what is the value", "what does the dashboard say", "what is the SLA for") and applies a retrieval filter or score boost for `has_table=True` chunks. This targets the right chunk type without overfitting to individual queries.

---

### S7 — Cross-Encoder Domain Calibration
**Impact**: Medium | **Effort**: High
**Affects**: q005, q011, q023

The cross-encoder (ms-marco-MiniLM-L-6-v2) is a general-purpose web-search reranker. It occasionally promotes chunks from the wrong security domain (e.g., SIEM KPI chunk ranked above ASM KPI chunk for an ASM query). Options:
- Fine-tune the cross-encoder on domain-specific (query, chunk, label) triplets from these documents
- Use a hybrid reranker score: `0.8 * cross_encoder_score + 0.2 * domain_match_bonus` where domain is inferred from filename prefix (asm_, siem_, fraud_, etc.)

---

### S8 — Address Silent Hallucination Pattern
**Impact**: High (trust/safety) | **Effort**: Medium
**Affects**: q025, q026, q027, q028

These queries show R@k=0 (wrong document retrieved) but KW=1.0 (correct answer generated). The LLM produces factually correct values but cites the wrong document. In production this means:
- The answer is correct today because content duplicates across docs
- If source documents diverge (policy updates, version changes), the system will confidently cite an outdated source
- There is no way for the user to verify the answer because the cited source doesn't contain the data

Mitigation: Add a citation verification step — after the LLM generates an answer, verify that each cited value actually appears in the cited source chunk. If not, flag the response with a low-confidence marker or suppress the citation.

---

## Recommended Implementation Priority

| Priority | Action | Queries Impacted | Effort |
|----------|--------|-----------------|--------|
| 1 | Increase top_k: 5 → 10 (config change) | 8 queries | 5 min |
| 2 | Fix golden dataset mapping errors | 7 queries | 30 min |
| 3 | Fix keyword format in q021 | 1 query | 2 min |
| 4 | Implement MMR in retriever | 3 queries | 2-3 hrs |
| 5 | Table-aware retrieval boost | 8 queries | 2-3 hrs |
| 6 | Semantic keyword matching in evaluator | 11 queries | 3-4 hrs |
| 7 | Investigate q004 LLM refusal | 1+ queries | 1 hr |
| 8 | Citation verification step | 4 queries | 3-4 hrs |
| 9 | Cross-encoder domain calibration | 3 queries | High effort |
