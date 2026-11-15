# MarketMind Intelligence

**AI-powered customer feedback analysis, demand forecasting, and inventory decision support**

**Live website:** https://marketmind-intelligence-zn65.onrender.com
**GitHub repository:** https://github.com/Arshiabali/MarketMind-Intelligence

## Overview

MarketMind Intelligence is a Python-based, interactive business intelligence prototype combining customer review analysis with demand and stock-planning tools. The Gradio interface provides four tabs for exploring review summaries, analyzing new review text, estimating seven-day product demand, and generating stock replenishment recommendations.

## Features

- **Business Dashboard:** Displays review and product statistics from the saved dataset.
- **Customer Intelligence:** Analyzes a customer review to predict sentiment and identify a review theme.
- **Demand Intelligence:** Estimates product demand for the next seven days using a trained model and scenario inputs.
- **Inventory Intelligence:** Estimates reorder points, safety stock, and suggested replenishment quantities.

## Dataset and Models

The customer-review analysis uses an Amazon Beauty review dataset. The exported dataset contains 2,999 reviews across 2,477 products. Review text is processed using vectorization and machine-learning models for sentiment and theme analysis. Demand forecasting uses a trained model, while inventory recommendations use demand estimates and stock-planning calculations.

**Note:** Demand and inventory examples are based on simulated business data. They demonstrate the workflow rather than verified real-world sales forecasts.

## Technologies Used

Python, Gradio, pandas, NumPy, scikit-learn, joblib, GitHub, and Render.

## Run Locally

```bash
git clone https://github.com/Arshiabali/MarketMind-Intelligence.git
cd MarketMind-Intelligence
pip install -r requirements.txt
python app.py
```

The app expects the trained model files in `models/` and the business CSV in `data/`.

## Deployment

The application is hosted on Render and uses Gradio for its interface. The live site can be opened here:

https://marketmind-intelligence.onrender.com

Render's free instance may take some time to start after inactivity.

## Limitations and Future Work

The dashboard uses an exported dataset snapshot, and the demand and inventory data are simulated. Future improvements could include real sales data, a connected database, and periodic model updates.
