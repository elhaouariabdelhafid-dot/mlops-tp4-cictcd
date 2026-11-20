import argparse

from mlflow import MlflowClient

from src.data import load_params


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--approve",
        action="store_true",
        help="Simule une approbation humaine"
    )

    args = parser.parse_args()

    params = load_params()
    model_name = params["training"]["model_name"]

    client = MlflowClient(
        tracking_uri="http://127.0.0.1:5000"
    )

    # Recuperer la version candidate
    candidate = client.get_model_version_by_alias(
        model_name,
        "candidate"
    )

    # Verifier le statut de validation
    status = candidate.tags.get("validation_status")

    print("=== PROMOTION DU MODELE ===")
    print("Modele :", model_name)
    print("Candidate : version", candidate.version)
    print("Validation :", status)

    if status != "passed":
        raise RuntimeError(
            "Le modele n'a pas passe les Quality Gates."
        )

    # Sans approbation, aucune promotion
    if not args.approve:
        print("Aucune approbation humaine.")
        print("Le modele reste candidate.")
        return

    # Promotion vers champion
    client.set_registered_model_alias(
        model_name,
        "champion",
        candidate.version
    )

    client.set_model_version_tag(
        model_name,
        candidate.version,
        "approval_status",
        "approved"
    )

    print("Approbation accordee.")
    print("Champion -> version", candidate.version)
    print("Promotion terminee avec succes.")


if __name__ == "__main__":
    main()
