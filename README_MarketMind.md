# MarketMind Intelligence

**AI-powered customer feedback analysis, demand forecasting, and inventory decision support**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46a2f1)](https://marketmind-intelligence.onrender.com) [![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/) [![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://www.gradio.app/)

**[Open the live MarketMind app](https://marketmind-intelligence.onrender.com)** · **[Source repository](https://github.com/Arshiabali/MarketMind-Intelligence)**

> **Project scope:** Sentiment and review-theme models were trained using Amazon Beauty customer reviews. The demand and inventory examples use **simulated business data**; their outputs are demonstrations, not validated forecasts of real-world sales.

## Overview

MarketMind Intelligence is a Python-based, interactive business intelligence prototype combining customer review analysis with demand and stock-planning tools. The Gradio interface provides four tabs for exploring review summaries, analyzing new review text, estimating seven-day product demand, and generating stock replenishment recommendations.

## Features

| Module | What it does |
| --- | --- |
| **Business Dashboard** | Displays exported review and product summary counts. |
| **Customer Intelligence** | Predicts review sentiment, assigns a learned review-theme cluster, and provides a short interpretation. |
| **Demand Intelligence** | Estimates next-seven-day unit demand from product history and editable scenario inputs. |
| **Inventory Intelligence** | Calculates estimated lead-time demand, safety stock, reorder point, stock status, and recommended order quantity. |

### Dataset snapshot

The exported deployment metadata reports **2,999 reviews**, **2,477 products**, **2,316 positive**, **295 neutral**, and **388 negative** reviews. These are counts from the exported dataset snapshot, not live user traffic.

## How it works

```text
Amazon Beauty reviews ──> Text vectorization ──> Sentiment model ──> Sentiment output
                  └─────> Theme vectorization ──> K-Means clusters ──> Review themes

Simulated product sales / scenario inputs ──> Engineered features
                                           └──> Trained demand model ──> 7-day forecast
                                                                        └──> Inventory rules
                                                                             └──> Restock advice
```

The application loads saved scikit-learn-compatible model artifacts using `joblib`. The sentiment and theme analyses operate on newly entered review text. The demand model uses historical lag, rolling sales, pricing, rating, feedback, promotion, date, and stock features. The inventory module then applies a lead-time and safety-stock heuristic to the estimated demand.

## Technology stack

- Python 3.11, Gradio
- pandas, NumPy
- scikit-learn, joblib
- GitHub for source control and Render for deployment

## Run locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/Arshiabali/MarketMind-Intelligence.git
cd MarketMind-Intelligence
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio. **Use only model artifacts you trust**; loading untrusted pickle/joblib files can execute code.

## Required project layout

```text
MarketMind-Intelligence/
├── app.py
├── requirements.txt
├── render.yaml                 # optional Render configuration
├── models/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── theme_vectorizer.pkl
│   ├── kmeans_model.pkl
│   ├── demand_prediction_model.pkl
│   └── deployment_metadata.json
└── data/
    └── business_latest.csv
```

The model files and CSV must be in their **named folders**, not beside `app.py`. The training/export notebook is useful for reproducibility but is not required for routine inference after artifacts are saved.

## Deploy on Render

1. Push the required files and folders to GitHub.
2. Create a **Web Service** on [Render](https://dashboard.render.com/) and connect the repository.
3. Select a Python runtime and, where available, the **Free** instance.
4. Set build command: `pip install -r requirements.txt`.
5. Set start command: `python app.py`.
6. The app must bind to `0.0.0.0` and use Render's `PORT` environment variable; the packaged `app.py` is configured for this.
7. Deploy and test each tab at the live URL.

**Hosting note:** Render's free services may spin down after inactivity, and the first request after sleep can be slow. Continuous 24/7 uptime is not guaranteed on the free tier.

## Testing checklist

- [ ] Dashboard loads and exported counts appear.
- [ ] Customer Intelligence accepts a new review and returns sentiment and a theme.
- [ ] Demand Intelligence produces a numeric seven-day estimate for a product.
- [ ] Inventory Intelligence returns stock status and a recommendation table.
- [ ] App still works after a fresh Render deployment.

## Limitations and responsible use

- Review labels and theme clusters depend on the training data and can be inaccurate or biased.
- Topic clusters are keyword-based groupings, not verified causal explanations.
- Demand and inventory figures rely on **simulated** business data and are **not validated real-world sales forecasts**.
- The dashboard counts are a fixed exported snapshot; they do not update when users submit new reviews.
- No real-time business database, user accounts, or persistent review storage are implemented in this deployment.
- Model performance metrics should be reported only when supported by notebook evaluation outputs; no accuracy or forecast-error figure is claimed here.

## Future improvements

Real transactional data integration, out-of-sample evaluation, drift monitoring, scheduled retraining, persistent dashboards, and more robust uncertainty estimates.

## Author

Developed as a machine-learning and business-intelligence portfolio project.
