"""
Always use this package for importing config in other modules
"""

from argparse import Namespace

from cmipiu.src.config.hyperparams import add_hyperparameters

__all__ = [
    'config'
]

config = Namespace(use_pciat=False)
add_hyperparameters(config)
