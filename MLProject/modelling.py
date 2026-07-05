"""
modelling.py (versi untuk MLflow Project / Workflow CI - Kriteria 3)

Script ini dipanggil otomatis oleh `mlflow run .` sesuai definisi pada file
`MLProject`. Setiap kali workflow CI trigger (push ke branch main), model akan
dilatih ulang dan tracking-nya disimpan sebagai artefak lokal (folder mlruns)
yang kemudian di-commit balik ke repository oleh workflow.
"""

import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main(data_path: str):
    mlflow.set_experiment("Telco Customer Churn - CI")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="CI_RandomForest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy : {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="namadataset_preprocessing/Telco-Customer-Churn_preprocessing.csv",
    )
    args = parser.parse_args()
    main(args.data_path)
