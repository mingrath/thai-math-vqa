# Thai math visual question answering

SuperAI Engineer S6 — individual Kaggle hackathon.

- **Competition:** https://www.kaggle.com/competitions/super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen
- **Slug:** `super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen`

## Get the data
```bash
kaggle competitions download -c super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen -p data/
cd data && unzip -q '*.zip' && rm -f *.zip && cd ..
```

## Layout
```
data/         # kaggle data (gitignored)
notebooks/    # colab_thai_math_vqa.ipynb — end-to-end Colab pipeline
src/          # pipeline code: normalize, data, prompts, infer
run_pipeline.py  # CLI entrypoint (eval / predict)
submissions/  # generated submission csvs (gitignored)
```

## Pipeline (Colab Pro)
Open `notebooks/colab_thai_math_vqa.ipynb` in Colab (GPU: A100 40GB / L4 24GB) and run top
to bottom: install → clone → Kaggle data → eval on train → predict → submit.

Approach: open-weights VLM (default `Qwen/Qwen2.5-VL-7B-Instruct`, no API) served with
vLLM, **self-consistency** (sample N CoT paths, majority-vote on the normalized answer),
and an exact-match **normalization** layer (`src/normalize.py`) replicating the official
metric (Thai digits, strip units/`$`, expand `\frac`/`\sqrt`/`\pi`, canonical integer).

## Run locally / on a GPU box
```bash
pip install -r requirements.txt

# measure expected score on the train split (has gold answers)
python run_pipeline.py --mode eval --n 8 --save-eval eval_train.csv

# generate the submission
python run_pipeline.py --mode predict --n 8 --out submissions/sub.csv
```
Key flags: `--model`, `--n` (self-consistency samples), `--max-pixels`, `--tp` (#GPUs),
`--limit` (quick test), `--no-cot`.

## Submit
```bash
kaggle competitions submit -c super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen -f submissions/sub.csv -m "message"
```
Limit: **5 submissions total** — tune with `--mode eval` first.
