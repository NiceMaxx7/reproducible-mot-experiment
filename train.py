import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Подключаемся к серверу MLflow (из Docker Compose или локально)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("reproducible-mot-experiment")

# Загружаем датасет, отслеживаемый через DVC
df = pd.read_csv("data/dataset.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="rf_baseline_v1"):
    n_estimators = 100
    max_depth = 5
    random_state = 42

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("dataset_source", "data/dataset.csv (DVC)")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="weighted")
    acc = accuracy_score(y_val, y_pred)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Run completed! Accuracy: {acc:.4f}, F1-score: {f1:.4f}")