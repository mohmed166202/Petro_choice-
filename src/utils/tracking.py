"""
tracking.py - MLflow Experiment Tracking Manager
------------------------------------------------
Provides a self-contained local MLflow setup backed by SQLite,
ensuring offline reproducibility without external cloud accounts.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import mlflow


def setup_mlflow(
    experiment_name: str = "ppe_helmet_detection",
    db_path: str = "sqlite:///mlflow.db",
    artifact_location: Optional[str] = None
) -> str:
    """
    Initializes MLflow tracking URI and ensures experiment exists.
    
    Args:
        experiment_name (str): The name of the MLflow experiment.
        db_path (str): SQLite tracking URI.
        artifact_location (str, optional): Directory to store artifacts.
        
    Returns:
        str: The experiment ID.
    """
    mlflow.set_tracking_uri(db_path)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location or str(Path("mlruns").resolve())
        )
    else:
        exp_id = experiment.experiment_id
        
    mlflow.set_experiment(experiment_name)
    return exp_id


def log_run_metadata(
    run_id: str,
    changed_variable: str,
    seed: int,
    params: Dict[str, Any],
    git_commit: str = "unknown"
) -> None:
    """
    Standardizes schema logging for every experiment run.
    """
    mlflow.set_tag("run_id", run_id)
    mlflow.set_tag("changed_variable", changed_variable)
    mlflow.set_tag("seed", str(seed))
    mlflow.set_tag("git_commit", git_commit)
    
    # Flatten parameters if nested
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool)):
            mlflow.log_param(k, v)


if __name__ == "__main__":
    exp_id = setup_mlflow("test_experiment")
    print(f"MLflow initialized with SQLite backend. Experiment ID: {exp_id}")
