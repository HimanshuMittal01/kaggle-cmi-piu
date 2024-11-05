from enum import Enum

class FeatureEngineeringSet(Enum):
    Normal        = 'Normal'
    AutoencodedPQ = 'AutoencodedPQ'


class ModelLevel0(Enum):
    AutoencoderEnsemble = 'autoencoder_ensemble'
    PlainEnsemble       = 'plain_ensemble'
    NaiveEnsemble       = 'naive_ensemble'
