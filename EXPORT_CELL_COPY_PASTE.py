# RUN ONCE in your existing, already-run MarketMind Google Colab session.
# It uses the trained variables from the working notebook and downloads a ZIP.
import os, json, shutil, zipfile, joblib
from pathlib import Path
from google.colab import files

export_dir=Path('/content/marketmind_export')
(export_dir/'models').mkdir(parents=True,exist_ok=True)
(export_dir/'data').mkdir(parents=True,exist_ok=True)
model_names=['sentiment_model.pkl','tfidf_vectorizer.pkl','theme_vectorizer.pkl','kmeans_model.pkl','demand_prediction_model.pkl']
for name in model_names:
    src=Path(PROJECT_DIR)/'models'/name
    if not src.exists():raise FileNotFoundError(f'Missing {src}. Run training notebook cells first.')
    shutil.copy2(src,export_dir/'models'/name)

# Preserve the exact feature rows used by the trained demand model.
latest=business_ml_df.sort_values('date').groupby('product_id',as_index=False).tail(1)
latest[['product_id']+DEMAND_FEATURES].to_csv(export_dir/'data'/'business_latest.csv',index=False)
meta={'demand_features':list(DEMAND_FEATURES),
      'cluster_keywords':{str(k):list(v) for k,v in cluster_keywords.items()},
      'review_stats':{'total_reviews':int(len(clean_df)),
                      'total_products':int(clean_df['product_id'].nunique()),
                      'positive':int((clean_df['sentiment']=='Positive').sum()),
                      'neutral':int((clean_df['sentiment']=='Neutral').sum()),
                      'negative':int((clean_df['sentiment']=='Negative').sum())}}
(export_dir/'models'/'deployment_metadata.json').write_text(json.dumps(meta,indent=2))
zip_path=Path('/content/marketmind_trained_artifacts.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in export_dir.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(export_dir))
print('Exported files:',*[str(p.relative_to(export_dir)) for p in export_dir.rglob('*') if p.is_file()],sep='\n - ')
files.download(str(zip_path))
