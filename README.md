---
title: MarketMind Intelligence
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# MarketMind Intelligence

Gradio app for customer review sentiment, topic clusters, 7-day demand prediction and inventory recommendations. The **demand/inventory dataset is simulated**. This repository is the deployable app; the original Colab notebook is the training workflow.

## First-time setup (no VS Code or HTML)

1. Open your **working MarketMind Colab notebook**, run all cells, and **in the same runtime** paste the contents of `EXPORT_CELL_COPY_PASTE.py` into a new cell. Run it once. It downloads `marketmind_trained_artifacts.zip`.
2. Unzip that archive on your computer. It contains the `models/` and `data/` folders. **Never upload only the starter ZIP and expect it to work:** the starter ZIP does not contain your trained model binaries.
3. Create a GitHub repository and upload all files from this starter project, **plus the exported model files and CSV in the matching folders**. GitHub web upload supports folders via drag-and-drop. If any model is over GitHub's web upload limit, use Hugging Face Spaces Files UI instead or Git LFS.
4. At [huggingface.co/new-space](https://huggingface.co/new-space), create a Space with **Gradio SDK**. Upload the same files through the Space's **Files → Add file** interface, or configure GitHub-to-Space syncing separately. A GitHub repository by itself **does not** automatically deploy a Space.
5. Wait for the Space to build; open its public URL. After that, it loads trained artifacts on startup without retraining or rerunning Colab. Free Spaces may sleep when idle and wake on the next visit.

## Required layout

```
app.py
requirements.txt
README.md
models/
  sentiment_model.pkl
  tfidf_vectorizer.pkl
  theme_vectorizer.pkl
  kmeans_model.pkl
  demand_prediction_model.pkl
  deployment_metadata.json
data/
  business_latest.csv
```

## Notes

- The model files and CSV **must** come from your successful Colab training run; they are not included in the starter ZIP because they were not attached to this conversation.
- If you retrain with different scikit-learn versions, you may need to align dependency versions. The pickle/joblib files should only be loaded from a trusted source.
- Large model artifacts may need Git LFS or hosting directly in the Hugging Face Space.
- GitHub can contain the full training `.ipynb` as an optional project artifact. The Space launches `app.py`, not the notebook.
- If the Space fails to start, check **Space → Logs** for missing files or version mismatches.
