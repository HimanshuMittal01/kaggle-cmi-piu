"""
Main run file
"""

from pathlib import Path

from cmipiu.flows import (
    run_autoencoder_pipeline
)

if __name__ == '__main__':
    DATA_DIR = Path("input/child-mind-institute-problematic-internet-use/")

    output = run_autoencoder_pipeline(DATA_DIR)

    print(output)
