# Thai Math VQA — Task Spec

SuperAI Engineer S6 Individual Hackathon · Domain: **NLP / Vision-Language (VQA)**
Competition: https://www.kaggle.com/competitions/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen

## Goal
Given **one JPG image** of a Thai math problem (Thai text + diagram + math symbols), output **one short answer string**.

## Data
- `images/images/*.jpg` — problem images (`0.jpg`, `1.jpg`, …).
- **Total 700** problem images. **Split: Train 280 / Test 420.** Per item: 1 image → 1 answer.
- 9 source buckets (counts): 101→26, 102→27, 103→125, 104→134, 105→101, 116→105, 118→38, 120→30, 122→114.
- Pull: `kaggle competitions download -c super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen -p data/`

## Task & answer formats
- **Topics:** arithmetic · algebra · geometry · combinatorics · word problems.
- **Grades:** primary → lower-secondary → upper-secondary.
- **Answer can be:** integers, decimals, LaTeX, Thai-unit phrases, short Thai text.

| Format | Example |
|---|---|
| integer / decimal | `36` · `-2` · `2.8125` |
| number + Thai unit | `20 ตารางเซนติเมตร` · `30 องศา` |
| LaTeX | `$6\sqrt{3}$` · `$\frac{17}{10}$` |
| short Thai phrase | `ขาดทุน ร้อยละ 1` · `ข้อ (จ)` |

## Submission format
CSV: `id, answer` (one short answer per image).

## Metric
**Exact-match Accuracy (after normalization).** Public LB ≈ 34% of test set · Private LB ≈ 66% (decisive).

### Normalize pipeline (apply to both pred & gold)
- lowercase + strip whitespace
- Thai digits ๐–๙ → 0–9
- strip `$` and units (องศา, หน่วย, บาท, …)
- expand LaTeX: `\frac`, `\sqrt`, `\pi`
- strip `{ } \ ,` + canonical integer (`2.0` → `2`)

## ⚠️ Key prohibition
**No commercial VLM API.** Use only self-run **open-weights** VLM (e.g. Qwen2-VL / InternVL / Typhoon-Vision, run locally).

## Hackathon rules (common to all 5 tasks)
- ❌ No commercial / third-party API or LLM to generate the submission.
- ❌ No private datasets. ✅ GenAI allowed for coding. ✅ AutoML allowed. ✅ Internet search allowed.
- Compute: LANTA HPC, 100 SHr/person. Submission limit: 5/day per hackathon.
- Source briefing: `~/Downloads/SPAI SS6 Individual.pdf`.
