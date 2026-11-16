"""MarketMind Intelligence — deployable Gradio interface (trained artifacts required)."""
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
DATA = ROOT / 'data'
NEEDED = ['sentiment_model.pkl','tfidf_vectorizer.pkl','theme_vectorizer.pkl',
          'kmeans_model.pkl','demand_prediction_model.pkl']
missing = [str(MODELS / x) for x in NEEDED if not (MODELS / x).is_file()]
if not (DATA / 'business_latest.csv').is_file():
    missing.append(str(DATA / 'business_latest.csv'))
if missing:
    raise RuntimeError('Missing trained artifacts. Run export_for_deployment.ipynb ONCE in the original Colab session and upload the exported files: ' + ', '.join(missing))

sentiment_model = joblib.load(MODELS / 'sentiment_model.pkl')
sentiment_vectorizer = joblib.load(MODELS / 'tfidf_vectorizer.pkl')
topic_vectorizer = joblib.load(MODELS / 'theme_vectorizer.pkl')
topic_model = joblib.load(MODELS / 'kmeans_model.pkl')
demand_model = joblib.load(MODELS / 'demand_prediction_model.pkl')
metadata_path = MODELS / 'deployment_metadata.json'
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
DEMAND_FEATURES = metadata.get('demand_features', ['lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_7', 'rolling_14', 'rolling_30', 'sales_std_7', 'price', 'rating', 'negative_pct', 'lead_time_days', 'promotion', 'day_of_week', 'month', 'day_of_year', 'current_stock'])
cluster_terms = {int(k):v for k,v in metadata.get('cluster_keywords',{}).items()}
latest_data = pd.read_csv(DATA / 'business_latest.csv')
if latest_data.empty: raise RuntimeError('business_latest.csv is empty')
latest_data['product_id'] = latest_data['product_id'].astype(str)
products = sorted(latest_data['product_id'].unique().tolist())


THEME_PATTERNS = {
    "Product Quality": [
        r"\bquality\b", r"\bdefective\b",
        r"\bbroken\b", r"\bbad product\b",
        r"\bpoor product\b", r"\bdamaged product\b"
    ],
    "Delivery Issue": [
        r"\bdelivery\b", r"\bshipping\b",
        r"\blate\b", r"\bdelayed\b",
        r"\barrived late\b"
    ],
    "Packaging Issue": [
        r"\bpackaging\b", r"\bpackage\b",
        r"\bpacking\b", r"\bleaking\b",
        r"\bspilled\b"
    ],
    "Price / Value": [
        r"\bexpensive\b", r"\boverpriced\b",
        r"\bcheap\b", r"\bprice\b",
        r"\bvalue for money\b"
    ],
    "Skin Reaction": [
        r"\birritation\b", r"\brash\b",
        r"\ballergy\b", r"\bbreakout\b",
        r"\bacne\b", r"\bburning\b"
    ],
    "Positive Experience": [
        r"\blove\b", r"\bloved\b",
        r"\bamazing\b", r"\bexcellent\b",
        r"\bperfect\b", r"\bgreat product\b"
    ],
}

def extract_review_themes(review):
    text = review.lower()
    matched = []

    for theme, patterns in THEME_PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            matched.append(theme)

    return ", ".join(matched) if matched else "Other / Unspecified"


def customer_analysis(review):
    if not isinstance(review,str) or not review.strip():
        return 'Enter a review', '', 'Please enter customer feedback.'
    try:
        vec = sentiment_vectorizer.transform([review])
        sentiment = str(sentiment_model.predict(vec)[0])
        cluster = int(topic_model.predict(topic_vectorizer.transform([review]))[0])
        topic = ', '.join(cluster_terms.get(cluster,[])[:5]) or f'Cluster {cluster}'
        explanation = {'Positive':'Positive feedback; maintain product quality.',
                       'Negative':'Negative feedback; investigate potential product issues.',
                       'Neutral':'Mixed or neutral feedback.'}.get(sentiment,'Review analyzed.')
        return sentiment,topic,explanation
    except Exception as exc: return 'Error','',str(exc)


def make_features(product_id,price,rating,negative_pct,lead_time_days,promotion,current_stock=None):
    matches=latest_data.loc[latest_data['product_id']==str(product_id)]
    if matches.empty: raise ValueError('Select a valid product ID')
    row=matches.iloc[0]
    price=float(price);rating=float(rating);negative_pct=float(negative_pct);lead_time_days=float(lead_time_days)
    if not np.isfinite([price,rating,negative_pct,lead_time_days]).all():raise ValueError('Inputs must be finite numbers')
    if price<0 or not 1<=rating<=5 or not 0<=negative_pct<=100 or lead_time_days<0:
        raise ValueError('Price >= 0, rating 1–5, negative feedback 0–100%, lead time >= 0 required')
    features=row[DEMAND_FEATURES].to_dict()
    features.update(price=price,rating=rating,negative_pct=negative_pct,lead_time_days=lead_time_days,promotion=int(promotion))
    if current_stock is not None:
        stock=float(current_stock)
        if not np.isfinite(stock) or stock<0:raise ValueError('Stock must be a nonnegative finite number')
        features['current_stock']=stock
    return features


def forecast(features):
    X=pd.DataFrame([{k:features[k] for k in DEMAND_FEATURES}])
    return max(0.,float(demand_model.predict(X)[0]))


def demand_prediction(product_id,price,rating,negative_pct,lead_time_days,promotion):
    try:
        result=forecast(make_features(product_id,price,rating,negative_pct,lead_time_days,promotion))
        return f'{result:.1f} units / next 7 days','Forecast based on simulated historical sales and product features.'
    except Exception as exc:return 'Error',str(exc)


def inventory_prediction(product_id,stock,price,rating,negative_pct,lead_time_days,promotion):
    try:
        f=make_features(product_id,price,rating,negative_pct,lead_time_days,promotion,stock)
        predicted=forecast(f);daily=predicted/7
        safety=max(0.,float(f.get('sales_std_7',0)))*np.sqrt(float(f['lead_time_days']))*1.65
        lead_demand=daily*float(f['lead_time_days'])
        reorder=lead_demand+safety
        order=max(0,int(np.ceil(predicted+safety-float(f['current_stock']))))
        status='URGENT RESTOCK' if float(stock)<=0.5*reorder else 'RESTOCK SOON' if float(stock)<reorder else 'STOCK HEALTHY'
        values={'Product':product_id,'Predicted 7-Day Demand':round(predicted,2),
                'Average Daily Demand':round(daily,2),'Lead-Time Demand':round(lead_demand,2),
                'Safety Stock':round(safety,2),'Reorder Point':round(reorder,2),
                'Current Stock':float(stock),'Recommended Order':order,'Stock Status':status}
        return status,pd.DataFrame([{'Metric':k,'Value':v} for k,v in values.items()])
    except Exception as exc:return 'Error',pd.DataFrame([{'Metric':'Error','Value':str(exc)}])


def dashboard():
    stats=metadata.get('review_stats',{})
    return f"""# MarketMind Business Intelligence Dashboard

**Total Reviews:** {stats.get('total_reviews','Not exported')}  
**Products Analyzed:** {stats.get('total_products','Not exported')}  
**Positive Reviews:** {stats.get('positive','Not exported')}  
**Neutral Reviews:** {stats.get('neutral','Not exported')}  
**Negative Reviews:** {stats.get('negative','Not exported')}

Customer Feedback → Sentiment → Review Themes → Demand Forecasting → Inventory Recommendations

**Note:** Customer review models were trained using Amazon Beauty reviews. Business demand and inventory data are **simulated demonstrations**, not real sales forecasts.
"""

with gr.Blocks(title='MarketMind | Business Intelligence') as demo:
    gr.Markdown('# 📊 MarketMind — AI Business Intelligence')
    gr.Markdown('Customer sentiment, review themes, 7-day demand forecasts and inventory recommendations. **Business data is simulated.**')
    with gr.Tab('📊 Dashboard'):
        board=gr.Markdown(dashboard())
        gr.Button('Refresh Dashboard').click(dashboard,outputs=board)
    with gr.Tab('💬 Customer Intelligence'):
        review=gr.Textbox(label='Customer Review',lines=4,placeholder='This product is excellent!')
        analyze=gr.Button('Analyze Review',variant='primary')
        sent=gr.Textbox(label='Sentiment');topic=gr.Textbox(label='Review Themes');explain=gr.Textbox(label='Business Interpretation')
        analyze.click(customer_analysis,inputs=review,outputs=[sent,topic,explain])
    with gr.Tab('📈 Demand Intelligence'):
        prod=gr.Dropdown(choices=products,value=products[0],label='Product ID')
        with gr.Row():
            price=gr.Number(value=100,label='Price');rating=gr.Number(value=4,label='Rating (1–5)');negative=gr.Number(value=10,label='Negative Feedback (%)')
        with gr.Row():
            lead=gr.Number(value=7,label='Lead Time (days)');promo=gr.Dropdown(choices=[0,1],value=0,label='Promotion Active?')
        predicted=gr.Textbox(label='7-Day Demand Prediction');detail=gr.Textbox(label='Explanation')
        gr.Button('Predict Demand',variant='primary').click(demand_prediction,inputs=[prod,price,rating,negative,lead,promo],outputs=[predicted,detail])
    with gr.Tab('📦 Inventory Intelligence'):
        iprod=gr.Dropdown(choices=products,value=products[0],label='Product ID')
        with gr.Row():
            stock=gr.Number(value=100,label='Current Stock');iprice=gr.Number(value=100,label='Price');irating=gr.Number(value=4,label='Rating (1–5)')
        with gr.Row():
            ineg=gr.Number(value=10,label='Negative Feedback (%)');ilead=gr.Number(value=7,label='Lead Time (days)');ipromo=gr.Dropdown(choices=[0,1],value=0,label='Promotion Active?')
        status=gr.Textbox(label='Stock Status');table=gr.Dataframe(label='Inventory Recommendation',interactive=False)
        gr.Button('Generate Restock Recommendation',variant='primary').click(inventory_prediction,inputs=[iprod,stock,iprice,irating,ineg,ilead,ipromo],outputs=[status,table])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=False,
        show_error=True
    )

