import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.data import load_params
from src.train import train


def main():
    params = load_params()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(
        params["training"]["experiment_name"]
    )

    model_name = params["training"]["model_name"]

    with mlflow.start_run(run_name="sentiment-candidate") as run:
        model, metrics = train()

        mlflow.log_params({
            "C": params["model"]["C"],
            "max_features": params["model"]["max_features"],
            "ngram_range": str(
                params["model"]["ngram_range"]
            ),
        })

        mlflow.log_metrics(metrics)

        print("Accuracy :", metrics["accuracy"])
        print("F1-score :", metrics["f1"])

        # Vérification des Quality Gates de performance
        accuracy_ok = (
            metrics["accuracy"]
            >= params["quality_gate"]["min_accuracy"]
        )

        f1_ok = (
            metrics["f1"]
            >= params["quality_gate"]["min_f1"]
        )

        if not (accuracy_ok and f1_ok):
            mlflow.set_tag("validation_status", "failed")
            raise RuntimeError(
                "Quality Gates non respectees : "
                "modele non enregistre."
            )

        mlflow.set_tag("validation_status", "passed")

        # Enregistrement du modèle dans MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name,
        )

        client = MlflowClient(tracking_uri=tracking_uri)

        version = model_info.registered_model_version

        if version is None:
            versions = client.search_model_versions(
                f"name='{model_name}'"
            )
            matches = [
                v for v in versions
                if v.run_id == run.info.run_id
            ]
            if not matches:
                raise RuntimeError(
                    "Version enregistree introuvable."
                )
            version = max(
                matches, key=lambda v: int(v.version)
            ).version

        client.set_model_version_tag(
            model_name, str(version),
            "validation_status", "passed"
        )

        client.set_model_version_tag(
            model_name, str(version),
            "use_case", "sentiment"
        )

        client.set_registered_model_tag(
            model_name,
            "task",
            "text-classification"
        )

        client.set_registered_model_alias(
            model_name,
            "candidate",
            str(version)
        )

        print("\n=== MLFLOW REGISTRY ===")
        print("Modele :", model_name)
        print("Version :", version)
        print("Alias : candidate")
        print("Validation : passed")
        print("Run ID :", run.info.run_id)


if __name__ == "__main__":
    main()
