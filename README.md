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

Insights:
- Using Total_PCIAT as target gives better CV score than using sii however it seems to overfit and hence using sii is performing better on LB
- Using Fitness Duration theoritically sounds good but looks like only test mins would influence it.
- Binning CGAS score is good because there are peaks at multiple of 5.


Issues:
- Print statements in parallel
- Colorful metaflow output
- Progress Bar


You may add following line in `metaflow/cli.py` inside `start()` function:
```py
ctx.color = True
```

This will enable colors in printing. The problem is that because we're running flow programmatically, metaflow log output is piped to some other program. This makes `stream.isatty()` False i.e. stream is not connected to interactive terminal. Now, in `click` package logic is that it will strip out color codes from the message if stream is not connected to tty or color is not forced. In metaflow, there is no option to force colors from outside. Monkey patching also didn't work for me (you may give it a try). Hence, above temporary fix to solve this problem.


support stdlibrary logging https://github.com/Netflix/metaflow/issues/180

Adding CLI available options in `tune()` command

Use end step for splitting into train and valid indices in prepare training data flow
