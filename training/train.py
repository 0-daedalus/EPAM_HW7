"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# Comment this lines if you have problems with MLFlow installation
import mlflow

mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf["general"]["data_dir"])
MODEL_DIR = get_project_dir(conf["general"]["models_dir"])
TRAIN_PATH = os.path.join(DATA_DIR, conf["train"]["table_name"])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_file",
    help="Specify inference data file",
    default=conf["train"]["table_name"],
)
parser.add_argument("--model_path", help="Specify the path for the output model")


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 3):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)  # Adjust to have three output neurons

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DataProcessor:
    def __init__(self) -> None:
        pass

    def prepare_data(self) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)


class Training:
    def __init__(self) -> None:
        self.model = NeuralNetwork()
        self.criterion = (
            nn.CrossEntropyLoss()
        )  # Change to multi-class classification loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def run_training(
        self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33
    ) -> None:
        logging.info("Running training...")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        start_time = time.time()
        self.train(train_loader)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        self.test(test_loader)
        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")
        return train_test_split(
            df[
                [
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ]
            ],
            df["y"],
            test_size=test_size,
            random_state=conf["general"]["random_state"],
        )

    def train(self, train_loader) -> None:
        logging.info("Training the model...")

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        num_epochs = conf["train"]["num_epochs"]
        for epoch in range(num_epochs):
            for inp, labels in train_loader:
                optimizer.zero_grad()

                outputs = self.model.forward(inp)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    def test(self, test_loader) -> float:
        logging.info("Testing the model...")
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inp, labels in test_loader:
                outputs = self.model(inp)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.numpy())
                y_true.extend(labels.numpy())

        class_report = classification_report(y_true, y_pred)

        logging.info(f"Classification Report: {class_report}")
        return class_report

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(
                MODEL_DIR,
                datetime.now().strftime(conf["general"]["datetime_format"]) + ".pickle",
            )
        else:
            path = os.path.join(MODEL_DIR, path)

        torch.save(self.model.state_dict(), path)


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data()
    tr.run_training(df, test_size=conf["train"]["test_size"])


if __name__ == "__main__":
    main()
