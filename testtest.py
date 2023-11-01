import yaml

from e2.utils.config_validations import ModelConfig

config_path = "e2/model_training/model_config.yml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_config = ModelConfig(**config["model"])

model_config.hyperparameters

validated_config = model_config.model_dump()
validated_config["hyperparameters"]
