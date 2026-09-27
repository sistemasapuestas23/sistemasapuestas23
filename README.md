# Football Prediction Engine (TheStatsAPI)

Production-ready end-to-end football match prediction system.

## Features
- TheStatsAPI ingestion (player stats, odds from 5 bookmakers, 10-year history, xG)
- Strict temporal TimeSeriesSplit (no data leakage)
- 6 base models: Dixon-Coles Poisson, XGBoost, CatBoost, PyTorch DNN, LSTM, LSTM-Momentum
- Stacking meta-ensemble
- Full rate-limiting + exponential backoff
- GitHub Actions CI ready

## Quick Start

```bash
# 1. Clone / unzip and enter directory
cd football_prediction_engine

# 2. Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and set THESTATSAPI_API_KEY=your_token

# 5. Train (example – replace IDs with real ones from TheStatsAPI)
python main.py --mode train \
  --competition_id comp_3039 \
  --season_ids sn_6125938 sn_6125937

# 6. Predict upcoming matches
python main.py --mode predict \
  --competition_id comp_3039 \
  --date_from 2026-09-27 \
  --date_to 2026-10-04
```

## GitHub Actions
- Push to `main` runs lint + import checks
- Workflow dispatch can run dry-run with secret `THESTATSAPI_API_KEY`

## Directory Layout
```
football_prediction_engine/
├── config/settings.py
├── data_loader/
├── feature_engineering/
├── models/
├── training/
├── inference/
├── pipeline/
├── utils/
├── main.py
├── requirements.txt
├── .env.example
└── .github/workflows/ci.yml
```

## License
MIT – use freely for research and production.
