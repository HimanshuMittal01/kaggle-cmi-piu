# kaggle-cmi-piu
My submission for Kaggle competition Child Mind Institute — Problematic Internet Use

Tech stack
- pipx
- polars
- plotly
- optuna
- lightgbm
- mkdocs
- ruff

Insights:
- Using Total_PCIAT as target gives better CV score than using sii however it seems to overfit and hence using sii is performing better on LB
- Using Fitness Duration theoritically sounds good but looks like only test mins would influence it.
- Binning CGAS score is good because there are peaks at multiple of 5.
