# kaggle-cmi-piu
My submission for Kaggle competition Child Mind Institute — Problematic Internet Use

Tech stack
- uv
- polars
- altair
- optuna
- lightgbm
- mkdocs
- ruff
- metaflow
- typer
- tqdm

Insights:
- Using Total_PCIAT as target gives better CV score than using sii however it seems to overfit and hence using sii is performing better on LB
- Using Fitness Duration theoritically sounds good but looks like only test mins would influence it.
- Binning CGAS score is good because there are peaks at multiple of 5.

Input --> Preprocess pipeline -> Train model pipeline --> Inference pipeline -> Must return ID

Feature importance --> Trained model
Analysis pipeline --> Predictions and Input

Issues:
- Print statements in parallel
- Logging
    - Color issue in print 'python file run' vs 'python main.py command'
    - Progress Bar
