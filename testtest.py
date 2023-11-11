import yaml

from e2.data_ops import DatasetIO, Preprocessor
from e2.model_training import TrainModel
from e2.utils.config_validations import ModelsConfig

config_path = "e2/model_training/model_config.yml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_config = ModelsConfig(**config["model"])

validated_config = model_config.model_dump()
validated_config["parameters_model"]


dataset_path = "datasets/CreditScoreTrain.csv"
dataset = DatasetIO(dataset_path)

data = dataset.read_data()

preprocessor = Preprocessor(data)
X_train, X_test, y_train, y_test = preprocessor.clean_and_split_data(test_size=0.2)
pipeline = preprocessor.create_processing_pipeline()

# preprocessor._process_numeric_features()
trainer = TrainModel("e2/model_training/model_config.yml")
trainer.train_model(X_train, y_train, pipeline)
