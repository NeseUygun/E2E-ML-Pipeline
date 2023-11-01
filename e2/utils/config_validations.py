from pydantic import BaseModel


class ModelParameters(BaseModel):
    n_estimators: int
    max_depth: int
    random_state: int


class ModelConfig(BaseModel):
    name: str
    hyperparameters: ModelParameters
