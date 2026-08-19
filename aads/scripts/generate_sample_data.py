"""
Sample Dataset Generator for AADS demo and validation runs.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_churn_dataset(output_path: Path, n_samples: int = 500) -> Path:
    """Generate a realistic customer churn dataset with missingness and categorical features."""
    np.random.seed(42)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 15, size=n_samples)
    total_charges = np.clip(total_charges, 20.0, None)

    contract_types = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_methods = np.random.choice(["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], size=n_samples)
    internet_service = np.random.choice(["DSL", "Fiber Optic", "No"], size=n_samples, p=[0.4, 0.45, 0.15])
    tech_support = np.random.choice(["Yes", "No", "No internet service"], size=n_samples)

    # Churn probability heuristic
    churn_prob = (
        0.35
        + (contract_types == "Month-to-Month") * 0.25
        + (monthly_charges > 80) * 0.15
        - (tenure > 36) * 0.25
        - (tech_support == "Yes") * 0.15
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.90)
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(n_samples)],
        "tenure_months": tenure,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": np.round(total_charges, 2),
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "churn": churn,
    })

    # Inject realistic quirks: a few missing values & placeholder nulls
    df.loc[np.random.choice(n_samples, 12, replace=False), "total_charges"] = np.nan
    df.loc[np.random.choice(n_samples, 5, replace=False), "payment_method"] = "?"

    df.to_csv(output_path, index=False)
    print(f"Generated sample churn dataset at {output_path} ({n_samples} rows)")
    return output_path


if __name__ == "__main__":
    generate_churn_dataset(Path("data/sample_churn.csv"))
