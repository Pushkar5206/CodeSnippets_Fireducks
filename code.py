import fireducks as fd
import pandas as pd

# Load dataset using FireDucks
dataset = fd.read_csv('large_dataset.csv', optimize=True)

# Perform quick transformations
dataset = fd.transform(dataset, normalize=True, drop_null=True)

# Split for AI model training
train_data, test_data = fd.split(dataset, ratio=0.8)

print("FireDucks made data processing effortless!")
