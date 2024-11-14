# Kaggle CMI PIU
My submission for Kaggle competition Child Mind Institute — Problematic Internet Use. This repo packs more than that. This is developed on following tech stack:

![Tech Stack](static/My%20Upgraded%20Tech%20Stack.png "Tech Stack")

### Insights from the experiments
- Using Total_PCIAT as target gives better CV score than using sii however it seems to overfit and hence using sii is performing better on LB
- Using Fitness Duration theoritically sounds good but looks like only test mins would influence it.
- Binning CGAS score is good because there are peaks at multiple of 5.

<!-- Adding CLI available options in `tune()` command

Use end step for splitting into train and valid indices in prepare training data flow -->


<details>
<summary>How to get metaflow terminal colors?</summary>

Add following line in `venv/lib/python3.10/site-packages/metaflow/cli.py` inside `start()` function (your metaflow installed path could be different):
```py
ctx.color = True
```

This will enable colors in printing. The problem is that because we're running flow programmatically, metaflow log output is piped to some other program. This makes `stream.isatty()` False i.e. stream is not connected to interactive terminal. In `click` package logic is that it will strip out color codes from the message if stream is not connected to tty or color is not forced. In metaflow, there is no option to force colors from outside. Monkey patching also didn't work for me (you may give it a try). Hence, above temporary fix to solve this issue.
</details>

<details>
<summary>How to work with logging and CLI options?</summary>

Here, I use `typer` to define CLI options, `hydra` to define config between typer and metaflow, `logging` to enhance metaflow with file-handler-like features to store logs at separate place so this gives us full access to all features while keeping the code well-defined.

Metaflow logging might not be great for research experiments. There is an open issue for it: [support stdlibrary logging](https://github.com/Netflix/metaflow/issues/180)

However, I've not been able to identify alternative for logging progress bar (tqdm, rich) to a external file. There is an open issue for it: [tqdm integration](https://github.com/Netflix/metaflow/issues/32)

</details>
