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
notebooks/    # EDA + experiments
src/          # reusable pipeline code
submissions/  # generated submission csvs (gitignored)
```

## Submit
```bash
kaggle competitions submit -c super-ai-engineer-ss-6-individual-test-thai-math-vqa-challen -f submissions/sub.csv -m "message"
```
