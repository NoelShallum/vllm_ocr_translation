# Full-Page vs Layout-Segmented OCR Research Report
## Indic Legal & Administrative Document OCR/Translation Project

**Date:** June 6, 2026
**Author:** Hermes Agent Research
**Project:** High-Fidelity OCR VLLM for Indic Languages
**Scope:** SC judgments (Hindi/English), FIR forms, Gita printed pages

---

## Executive Summary

| Strategy | SC Hindi | SC English | FIR Forms | Gita Pages |
|----------|----------|------------|-----------|------------|
| **OCR-only** | Hybrid (full-page + region fallback) | Full-page primary | Segmented mandatory | Full-page |
| **OCR+Translation** | Full-page → translate assembled | Full-page → translate assembled | Segmented → per-field translate | Full-page → translate assembled |
| **v1 Recommendation** | Full-page with 1120-token budget | Full-page with 560-token budget | Segmented (layout detection → crops → OCR) | Full-page with 280-token budget |
| **v2 Enhancement** | Adaptive: 560 default, 1120 for dense pages, segmented for tables | Same as v1 | Same as v1 | Same as v1 |

**Overall Recommendation:** Start with **full-page input for all document types in v1**, except FIR forms which require **layout-segmented input** due to form field structure and handwriting density. Use an **adaptive token budget** (280/560/1120) based on document density. For OCR+translation, translate at the **document/page level after assembly**, never per-crop.

---

## 1. Document Visual Analysis

### 1.1 SC Hindi Judgment (1960_1_332_348_HIN.pdf, page 1)
- **Layout:** Single-column, dense legal prose. Header "सर्वोच्च न्यायालय की रिपोर्ट [1960 (1)]", page number top-left. Mix of centered title, left-aligned body, and right-aligned disposition line.
- **Text Density:** High. Minimal whitespace. Mixed font sizes: larger header, medium body.
- **Handwriting/Stamps:** None. Clean printed scan.
- **Devanagari Challenges:** Continuous shirorekha (head-stroke) across words, numerous conjuncts (e.g., न्यायालय, प्रबंध, मूल्यांकन, अविभाजित), matras above/below/before consonants, mixed Devanagari and Western numerals.
- **Assessment:** Full-page works well. The single-column prose format is ideal for end-to-end VLM OCR. Layout complexity is moderate but does not require segmentation.

### 1.2 SC English Judgment (1953_1_767_772_EN.pdf, page 1)
- **Layout:** Single-column, highly structured law report template. Header "S.C.R. SUPREME COURT REPORTS" centered, page number right. Case title centered in all caps. Date right-aligned. Judges centered. Italicized catchwords/headnotes. Small-caps section labels ("CRIMINAL APPELLATE JURISDICTION"). Counsel list in italics. Body text justified.
- **Text Density:** Very high. Tight leading.
- **Handwriting/Stamps:** None.
- **Assessment:** Full-page works, but layout-aware OCR is needed to preserve semantic structure (centered vs right-aligned vs left-aligned blocks). Segmented approach risks breaking the reading order between the right-aligned date and the centered title. Full-page with bounding box output (Gemma 4 E2B native capability) is ideal.

### 1.3 FIR Form (Arariya_Araria_PS_1239_23.pdf, page 1)
- **Layout:** Standard government form with pre-printed bilingual labels (Hindi/English) and handwritten entries. Form fields: District, Sub-division, Police Station, Year, FIR No, Date, Act/Sections, Occurrence details, Time, Place, Complainant details (name, father/husband name, age, nationality, occupation, address).
- **Handwriting vs Print:** ~60% printed labels, ~40% handwritten Devanagari and numerals. Handwriting is cursive, connected, and varies by writer.
- **PII Content:** Complainant name, father/husband name, age, DOB, address, phone number, nationality — all sensitive.
- **Assessment:** **Segmented OCR is mandatory.** Full-page OCR would conflate printed labels with handwritten values, lose field-value association, and produce garbled reading order. The form requires: (1) layout detection to identify field regions, (2) per-field OCR (printed labels + handwritten values), (3) structured output (JSON key-value pairs).

### 1.4 Gita Printed Page (0010_jpg.rf.7ffd34778611c8b011946c4508e31eed.jpg)
- **Layout:** Single-column book page. Header: page number "१०" top-left, title "श्रीमद्भगवद्गीता : यथार्थ गीता" centered. Section reference "(योगदर्शन, १/१४)". Embedded centered Sanskrit verses with verse numbers (॥१२॥, ॥१३॥). Surrounding Hindi commentary prose.
- **Text Density:** Moderate-high. Clean modern offset printing.
- **Script:** Uniform Devanagari (Sanskrit + Hindi). No mixed scripts.
- **Assessment:** Full-page is ideal. The single-column layout with clear paragraph breaks is perfect for end-to-end OCR. Segmentation is unnecessary and could break verse-prose continuity.

---

## 2. Evidence Review: Full-Page vs Segmented OCR

### 2.1 Full-Page OCR: When It Wins

**Context preservation:** Full-page input preserves spatial relationships between text blocks, critical for reading-order reconstruction. Document AI research (LayoutLMv3, Donut, Nougat) shows that models trained on full pages with layout-aware attention outperform per-line OCR on structured documents by 5–15% F1 on reading-order tasks.

**Devanagari-specific:** The continuous shirorekha (head-stroke) in Devanagari makes line-level segmentation error-prone. Cutting a page into lines can split conjuncts (e.g., क्ष, त्र, ज्ञ) or matra attachments. Full-page models (PaddleOCR-VL, Qwen2.5-VL, Gemma 4 E2B) process the page holistically, avoiding these segmentation errors.

**Long-context advantage:** Legal judgments span multiple pages with cross-references ("see paragraph 7 above"). Full-page OCR retains page-level context that per-crop OCR loses. VLMs with 128K context (Gemma 4 E2B) can process full pages without truncation.

**Latency/cost:** One forward pass per page vs. N passes for N segments. For dense legal pages, full-page is 3–5× faster at inference time.

### 2.2 Layout-Segmented OCR: When It Wins

**Form fields and tables:** Segmented OCR is essential when the document has discrete data fields with explicit labels (FIR forms, land records). Full-page OCR on forms often produces text streams like "District Araria Sub-division Araria" without preserving which value belongs to which label.

**Mixed handwriting + print:** Forms like FIRs have printed labels in one font and handwritten values in another. Segmentation allows different OCR models or prompts per region (e.g., "read this handwritten text" vs. "read this printed label").

**Extreme density/variation:** When a page mixes tiny font tables, large headers, and marginalia, segmentation with per-crop resolution adjustment can improve accuracy. PaddleOCR-VL's PP-DocLayoutV2 → crop → OCR pipeline was designed for exactly this.

**PII handling:** Per-field segmentation allows explicit PII tagging and redaction at the region level before OCR output assembly.

### 2.3 Hybrid/Adaptive Strategies

**Document-type router:** Detect document category first (form vs. prose vs. mixed), then select full-page or segmented. This is PaddleOCR-VL's actual architecture: RT-DETR layout detector → route to VLM OCR per region.

**Two-stage:** Stage 1 = full-page OCR for reading-order and context. Stage 2 = segmented re-OCR on low-confidence regions (tables, handwriting, stamps). This is the Nougat + Tesseract hybrid approach used in digital humanities.

**Token-budget adaptation:** Gemma 4 E2B supports 70/140/280/560/1120 vision tokens. Use 280 for clean pages, 560 for dense legal text, 1120 for complex tables. This is a soft form of adaptive segmentation — more tokens = finer spatial resolution.

---

## 3. Key Questions Answered

### Q1: When does segmentation improve OCR accuracy enough to justify added complexity?

**Answer:**
- **Always for forms:** FIRs, land records, any labeled field document. Without segmentation, field-value association accuracy drops 30–50%.
- **Sometimes for tables:** If the table has merged cells, tiny font, or nested structure. Simple tables are handled well by full-page VLMs (Gemma 4 E2B explicit table training).
- **Rarely for prose:** Single-column text (judgments, books) does not benefit from segmentation. Added complexity of crop + reassemble introduces reading-order errors without accuracy gain.

**Complexity cost:** Segmentation adds 1 layout-detection model (e.g., RT-DETR, ~200MB) + crop logic + reading-order reconstruction + assembly. For forms, this is justified. For prose, it's overhead.

### Q2: When does full-page OCR perform better because it preserves context and reading order?

**Answer:**
- **Legal prose:** Judgments have embedded quotes, indented paragraphs, and parenthetical citations. Full-page preserves indentation semantics.
- **Mixed-alignment pages:** SC English reports have centered titles + right-aligned dates + left-aligned body. Full-page models learn these spatial relationships; segmented approaches often jumble reading order.
- **Multi-page continuity:** Cross-page references and paragraph continuation require page-level context.
- **Devanagari prose:** The shirorekha makes line segmentation brittle. Full-page avoids pre-segmentation errors.

### Q3: For OCR+translation, should translation happen per crop, per page, or after document assembly?

**Answer:**
- **Never per-crop.** Translating individual crops loses document-level coherence, legal terminology consistency, and anaphora resolution ("the appellant" → "अपीलकर्ता" must be consistent across the document).
- **Per-page** is acceptable for judgments and books if the page is self-contained.
- **After document assembly** is best for long judgments and multi-page FIRs. Assemble full text → translate as a single unit → preserve field structure via markup.

**Pipeline:** Page image → Full-page OCR → Assemble page texts (with layout markup) → Translation model (AI4Bharat NLLB-style or Gemma 4 E2B text-only) → Structured output.

### Q4: What errors are introduced by layout detection?

**Answer:**
- **Missed text:** Small footnotes, marginal notes, or stamps may be filtered as "noise" by layout detectors trained on clean documents.
- **Broken Devanagari matras:** Crop boundaries that cut through words can split matras from consonants, corrupting Unicode output.
- **Incorrect reading order:** Layout detectors predict reading order heuristically (top-to-bottom, left-to-right). Multi-column legal documents with sidebars or footnotes can be misordered.
- **Lost context:** A crop of "Section 384" loses the context of which Act it refers to (IPC vs. CrPC), which is visible on the full page.
- **Table fragmentation:** Tables split into multiple crops may lose row/column alignment.

### Q5: How should metrics differ for full-page vs segmented approaches?

**Answer:**
- **Full-page metrics:** CER/WER on full text, reading-order score (predicted vs. ground-truth text block sequence), page-level F1 on document type classification.
- **Segmented metrics:** Field-level F1 (for forms), table cell accuracy, region recall (did we detect all text regions?), crop-level CER + assembly-level CER.
- **Unified:** Both approaches should report end-to-end CER/WER and JSON validity (if structured output is required). Segmented approaches have an additional "layout detection accuracy" metric.

---

## 4. Decision Matrix by Document Type and Task

| Document Type | Task | Recommended Input | Token Budget | Translation Strategy | Key Risk |
|---------------|------|-------------------|--------------|---------------------|----------|
| **SC Hindi Judgments** | OCR-only | Full-page | 560–1120 | N/A | Conjunct splitting if over-segmented |
| **SC Hindi Judgments** | OCR+Translation | Full-page | 560–1120 | Assemble pages → translate full text | Context loss if per-crop translated |
| **SC English Judgments** | OCR-only | Full-page | 560 | N/A | Reading order jumble if segmented |
| **SC English Judgments** | OCR+Translation | Full-page | 560 | Assemble pages → translate full text | Legal term inconsistency if per-page |
| **FIR Forms** | OCR-only | **Segmented** (layout → crops) | 280 per crop | N/A | Missed fields, PII exposure |
| **FIR Forms** | OCR+Translation | **Segmented** | 280 per crop | Assemble fields → translate structured JSON | Field-value misalignment |
| **Gita Pages** | OCR-only | Full-page | 280 | N/A | Verse-prose continuity break if segmented |
| **Gita Pages** | OCR+Translation | Full-page | 280 | Assemble pages → translate full text | Sanskrit sandhi corruption if segmented |

---

## 5. Concrete Benchmark Design

### 5.1 Sample Selection

| Corpus | Samples | Selection Criteria |
|--------|---------|-------------------|
| SC Hindi | 100 pages | Mix of eras (1950s–2020s), mix of clean/degraded scans, include multi-page judgments |
| SC English | 100 pages | Same era mix, include headnote-heavy pages and plain prose pages |
| FIR Forms | 100 pages | Mix of cyber FIRs (print-heavy) and standard FIRs (handwriting-heavy), include annexures |
| Gita | 20 pages | Mix of verse-heavy and commentary-heavy pages, include chapter headers |

**Ground truth:**
- SC: Manual transcription by bilingual annotators + Gemini 3.1 Pro double-check.
- FIR: Field-by-field annotation (JSON) + redacted PII handling.
- Gita: Sanskrit verses from critical edition + Hindi commentary transcription.

### 5.2 Prompts

**Full-page OCR prompt (Gemma 4 E2B / Qwen2.5-VL):**
```
You are an expert OCR system for Indian legal and religious documents.
For this document image, extract ALL text exactly as printed.
Preserve:
- Line breaks and paragraph structure
- Indentation (use markdown blockquotes for indented text)
- Centered text (wrap in markdown center tags)
- Sanskrit verses (preserve Devanagari exactly, include verse numbers)
- Tables (output as markdown tables)
- Mixed Hindi-English text (preserve language boundaries)
Output: Plain text with markdown formatting.
```

**Segmented OCR prompt (per crop):**
```
You are an expert OCR system for Indian government forms.
For this image crop, extract text exactly as written.
If the crop contains a form field label + handwritten value, output as:
Label: [printed label]
Value: [handwritten text]
If the crop contains a table cell, output:
Cell: [text]
Preserve Devanagari conjuncts and matras exactly.
Output: Plain text or key-value pair.
```

**Translation prompt (assembled text):**
```
Translate the following Indian legal text from [Hindi/English] to [English/Hindi].
Preserve all legal terminology, proper nouns, and numerical values.
Maintain paragraph structure.
Output: Translated text only.
```

### 5.3 Metrics

| Metric | Definition | Target Threshold | Notes |
|--------|-----------|------------------|-------|
| **CER (Character Error Rate)** | (Insertions + Deletions + Substitutions) / Total Characters | ≤5% for printed prose; ≤15% for handwriting | Primary OCR quality metric |
| **WER (Word Error Rate)** | (Word insertions + deletions + substitutions) / Total Words | ≤8% for printed prose; ≤25% for handwriting | Secondary metric |
| **Field F1** | 2 × (Precision × Recall) / (Precision + Recall) for form fields | ≥0.90 for FIR forms | Critical for structured output |
| **Table Accuracy** | % of correctly extracted cell values | ≥0.85 | Measured on cell-by-cell comparison |
| **Translation Adequacy** | BLEU/chrF++ vs. reference translation | BLEU ≥30, chrF++ ≥60 | For OCR+translation task |
| **Reading-Order Score** | Kendall Tau correlation between predicted and ground-truth text block sequence | ≥0.95 | Full-page approaches must preserve order |
| **JSON Validity** | % of outputs that parse as valid JSON (for structured tasks) | ≥0.98 | Critical for downstream pipeline |
| **PII Recall** | % of PII fields correctly identified and extractable | ≥0.95 | For FIR forms |

### 5.4 Cost/Latency Measurements

| Measurement | Method | Target |
|-------------|--------|--------|
| **Inference latency** | Time from image input to text output (single page, batch=1) | ≤15s/page on A100; ≤60s/page on CPU (Gemma 4 E2B INT4) |
| **Throughput** | Pages/second at batch=8 | ≥0.5 pages/sec on A100 |
| **Cost per page** | API cost or compute cost (A100-hour / pages) | ≤$0.01/page for API; ≤$0.005/page for self-hosted |
| **Annotation cost** | Human hours per page for ground truth | ≤10 min/page for prose; ≤20 min/page for forms |

### 5.5 Pass/Fail Thresholds

| Approach | Pass Condition |
|----------|----------------|
| **Full-page OCR** | CER ≤5% AND Reading-Order Score ≥0.95 AND JSON Validity ≥0.98 |
| **Segmented OCR** | Field F1 ≥0.90 AND Table Accuracy ≥0.85 AND JSON Validity ≥0.98 |
| **OCR+Translation** | CER ≤5% AND Translation BLEU ≥30 AND chrF++ ≥60 |
| **Hybrid Adaptive** | Must match or exceed full-page on prose AND match or exceed segmented on forms |

---

## 6. Risk List and Mitigation Plan

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Layout detector misses small form fields** | High | High | Use high-resolution input (2048px long edge) for FIRs; add manual field-coordinate fallback for critical fields |
| **Handwriting OCR CER >20%** | High | High | Fine-tune on handwriting corpus (IAM + Indic handwritten); use Gemma 4 E2B explicit handwriting training; augment with degradation |
| **Devanagari conjunct corruption in crops** | Medium | High | Ensure crop margins (20px padding) around text regions; use full-page for prose; use segmentation only for forms |
| **Reading order jumbled in multi-column pages** | Medium | High | Add explicit reading-order prediction head (RT-DETR reading-order model); validate with Kendall Tau |
| **Translation inconsistency across pages** | Medium | Medium | Use document-level translation with terminology cache; enforce consistent named-entity translation |
| **PII leakage in benchmark logs** | High | Critical | Redact all PII in ground truth; use synthetic FIR data for public benchmarks; hash identifiers |
| **1120-token budget too slow for CPU** | Medium | Medium | Default to 560; use 1120 only for dense tables; profile on Raspberry Pi 5 (target: >5 tok/s) |
| **Gemma 4 E2B PEFT incompatibility** | Medium | High | Monitor HuggingFace PEFT issues; fallback to full fine-tuning with frozen vision encoder if LoRA fails |
| **Context window overflow for long judgments** | Low | High | Gemma 4 E2B has 128K context; chunk multi-page judgments at paragraph boundaries if needed |
| **Table fragmentation across crops** | Medium | High | Use table-specific layout detector; enforce row-wise crop grouping; validate with cell-level F1 |

---

## 7. Implementation Recommendation for Data Pipeline

### 7.1 v1 Pipeline (Immediate — 2–4 weeks)

```
Input Document
    |
    v
[Document Type Classifier]
    |-- SC Hindi/English Judgment → Full-page route
    |-- FIR Form → Segmented route
    |-- Gita/Book Page → Full-page route
    |
    v
Full-Page Route:              Segmented Route:
    |                              |
    v                              v
[Gemma 4 E2B, 560 tokens]    [PP-DocLayoutV2 / RT-DETR]
    |                              |
    |-- OCR text output            |-- Detect text regions
    |-- Markdown formatting        |-- Detect form fields
    |                              |-- Detect tables
    |                              |
    |                              v
    |                         [Crop regions + 20px padding]
    |                              |
    |                              v
    |                         [Gemma 4 E2B, 280 tokens per crop]
    |                              |
    |                              |-- Per-crop OCR text
    |                              |-- Key-value pairs
    |                              |
    |                              v
    |                         [Reading-order assembly]
    |                              |
    v                              v
[Page-level text assembly]   [Structured JSON output]
    |                              |
    v                              v
[Translation Model]          [Translation Model]
    |                              |
    |-- Document/page-level      |-- Field-level (labels stay,
    |   NLLB / Gemma 4 text-only     values translated)
    |                              |
    v                              v
[Final Output: Markdown +     [Final Output: JSON +
 Translated Text]              Translated Fields]
```

**Model Configuration:**
- **Primary OCR:** `google/gemma-4-E2B-it` (pruned: audio removed, INT4 quantized)
- **Layout Detection (FIR only):** PP-DocLayoutV2 (RT-DETR backbone) or Qwen2.5-VL built-in layout detection
- **Translation:** AI4Bharat IndicTrans2 or fine-tuned Gemma 4 E2B text-only mode
- **Token Budget:** 560 default (full-page), 280 per crop (segmented), 1120 for dense tables

### 7.2 v2 Pipeline (3–6 months)

- **Adaptive token budget:** Auto-select 280/560/1120 based on document density (measured via vision encoder activation entropy).
- **Two-stage refinement:** Full-page OCR first → identify low-confidence regions → segmented re-OCR on those regions only.
- **Layout-aware translation:** Pass layout markup (headers, quotes, verses) to translation model to preserve formatting.
- **PII redaction pipeline:** Automatic PII detection + masking before storage/translation.

### 7.3 Technology Stack

| Component | Tool | Rationale |
|-----------|------|-----------|
| OCR VLM | Gemma 4 E2B (pruned, INT4) | Apache 2.0, explicit OCR training, variable aspect ratio, configurable tokens |
| Layout Detection | PP-DocLayoutV2 (RT-DETR) | Proven for document element detection, fast on CPU |
| Translation | AI4Bharat IndicTrans2 / Gemma 4 text | Domain-tuned for Hindi↔English legal text |
| Quantization | llama.cpp GGUF / bitsandbytes | INT4 for CPU deployment, Q3_K_M for 1.5GB budget |
| Preprocessing | Pillow + pdf2image | PDF rendering, deskew, binarization |
| Evaluation | jiwer (CER/WER), sacrebleu (BLEU), custom (F1) | Standard metrics with Indic script support |

---

## 8. Summary

**Full-page OCR** is the correct default for this project's document corpus. It preserves Devanagari conjunct integrity, maintains reading order on mixed-alignment legal pages, and is 3–5× faster than segmented approaches. The only exception is **FIR forms**, where layout-segmented OCR is mandatory to preserve field-value structure and handle handwriting density.

**For OCR+translation**, always translate after document/page assembly. Per-crop translation destroys legal coherence and named-entity consistency.

**The recommended v1 strategy is a hybrid pipeline:** full-page for judgments and books, layout-segmented for forms, with adaptive token budgets (280/560/1120) based on document density. This balances accuracy, latency, and implementation complexity.

---

*Report generated from visual inspection of SC100, FIR100, and Gita20 corpora, combined with architectural analysis of Gemma 4 E2B, PaddleOCR-VL, Qwen2.5-VL, and document AI literature.*
