from pykeen.datasets import FB15k237

# Load the dataset
dataset = FB15k237()

# Access the predefined splits
training = dataset.training
validation = dataset.validation
testing = dataset.testing

# Display dataset statistics
dataset.summarize()