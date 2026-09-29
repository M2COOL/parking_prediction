from .lstm import LSTMModel
from .gru import GRUModel


MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "gru": GRUModel,
}


def create_model(model_name, **kwargs):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]

    return model_class(**kwargs)