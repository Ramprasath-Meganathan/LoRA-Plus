"""Load experiment results, select configurations, and plot convergence."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    COMMON_CONFIG,
    DATASET_CONFIGS,
    get_output_dir,
)


def load_results(model_name, dataset_name):
    """Load the CSV of completed runs for a model and dataset."""
    root = get_output_dir(model_name, dataset_name)
    path = root / f"results_{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def select_best(model_name, dataset_name):
    """Choose the best standard-LoRA and LoRA+ screening configurations."""
    df = load_results(model_name, dataset_name)

    screening = df[
        (df["stage"] == "screening")
        & (df["model_key"] == model_name)
        & (df["dataset"] == dataset_name)
        & (df["epochs"] == COMMON_CONFIG["epochs"])
        & (df["batch_size"] == COMMON_CONFIG["batch_size"])
        & (df["seed"] == COMMON_CONFIG["screening_seed"])
        & (df["max_steps"] == -1)
    ].copy()

    screening = screening.drop_duplicates(
        subset=["method", "lam", "eta_A"],
        keep="last",
    )

    lora = screening[screening["method"] == "LoRA"]
    plus = screening[screening["method"] == "LoRA+"]

    if len(lora) != 3 or len(plus) != 6:
        raise ValueError(
            "Expected 3 LoRA and 6 LoRA+ screening runs; "
            f"found {len(lora)} and {len(plus)}"
        )

    metric = DATASET_CONFIGS[
        dataset_name
    ]["primary_metric"]

    primary_column = {
        "accuracy": "final_acc",
        "f1": "final_f1",
    }[metric]

    screening = screening.sort_values(
        by=[primary_column, "final_eval_loss"],
        ascending=[False, True],
    )

    best_lora = screening[
        screening["method"] == "LoRA"
    ].iloc[0]

    best_plus = screening[
        screening["method"] == "LoRA+"
    ].iloc[0]

    print("\nBest standard LoRA:")
    print(
        best_lora[
            ["eta_A", "eta_B", primary_column]
        ]
    )

    print("\nBest LoRA+:")
    print(
        best_plus[
            ["eta_A", "eta_B", "lam", primary_column]
        ]
    )

    return best_lora, best_plus


def get_final_runs(model_name, dataset_name):
    """Load all final-seed runs for the selected hyperparameters."""
    df = load_results(model_name, dataset_name)
    best_lora, best_plus = select_best(
        model_name, dataset_name
    )

    final = df[
        (df["stage"] == "final")
        & (df["model_key"] == model_name)
        & (df["dataset"] == dataset_name)
        & (df["epochs"] == COMMON_CONFIG["epochs"])
        & (df["batch_size"] == COMMON_CONFIG["batch_size"])
        & (df["max_steps"] == -1)
        & (df["seed"].isin(COMMON_CONFIG["seeds"]))
    ].copy()

    final = final[
        (
            (final["method"] == "LoRA")
            & (final["lam"] == 1)
            & np.isclose(
                final["eta_A"],
                best_lora["eta_A"]
            )
        )
        |
        (
            (final["method"] == "LoRA+")
            & (final["lam"] == best_plus["lam"])
            & np.isclose(
                final["eta_A"],
                best_plus["eta_A"]
            )
        )
    ].copy()

    final = final.drop_duplicates(
        subset=["method", "lam", "eta_A", "seed"],
        keep="last",
    )

    # Verify that both methods completed all expected seeds
    expected_seeds = set(COMMON_CONFIG["seeds"])

    for method in ["LoRA", "LoRA+"]:
        actual_seeds = set(
            final.loc[
                final["method"] == method,
                "seed"
            ].astype(int)
    )

    if actual_seeds != expected_seeds:
        raise ValueError(
            f"{method}: Expected seeds {expected_seeds}, "
            f"found {actual_seeds}"
        )

    return final


def summarize_final(model_name, dataset_name):
    """Aggregate final-run metrics across seeds and save a summary CSV."""
    final = get_final_runs(model_name, dataset_name)

    metrics = [
        "final_acc",
        "final_f1",
        "final_f1_macro",
        "final_eval_loss",
        "train_runtime_s",
        "peak_mem_gb",
        "trainable_params",
    ]

    summary = final.groupby(
        ["method", "lam", "eta_A"]
    )[metrics].agg(["mean", "std"])

    root = get_output_dir(model_name, dataset_name)

    summary.to_csv(root / "final_summary.csv")

    print("\nFinal results:")
    print(summary.to_string())

    return summary


def plot_convergence(model_name, dataset_name):
    """Plot mean validation convergence and seed variation for each method."""
    root = get_output_dir(model_name, dataset_name)
    final = get_final_runs(model_name, dataset_name)

    metric = (
        "eval_accuracy"
        if dataset_name == "sst2"
        else "eval_f1"
    )

    plt.figure(figsize=(9, 5))

    for method in ["LoRA", "LoRA+"]:
        selected = final[final["method"] == method]
        curves = []

        for _, row in selected.iterrows():
            run_name = (
                f"{model_name}_{dataset_name}_"
                f"lam{int(row['lam'])}_"
                f"lr{row['eta_A']:.8g}_"
                f"seed{int(row['seed'])}_"
                f"ep{int(row['epochs'])}_"
                f"bs{int(row['batch_size'])}_"
                f"steps-1_final"
            )

            path = root / "runs" / f"{run_name}.json"

            with open(path, "r") as f:
                record = json.load(f)

            points = {
                int(log["step"]): float(log[metric])
                for log in record["log_history"]
                if metric in log
            }

            curves.append(points)

        common_steps = sorted(
            set.intersection(
                *(set(c.keys()) for c in curves)
            )
        )

        if not common_steps:
            raise ValueError(
                f"No common evaluation steps for {method}"
            )

        values = np.array([
            [curve[step] for step in common_steps]
            for curve in curves
        ])

        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)

        plt.plot(
            common_steps,
            mean,
            label=method,
        )

        plt.fill_between(
            common_steps,
            mean - std,
            mean + std,
            alpha=0.15,
        )

    plt.xlabel("Optimizer steps")
    plt.ylabel(
        "Validation accuracy"
        if dataset_name == "sst2"
        else "Validation binary F1"
    )

    plt.title(
        f"{model_name.upper()} + {dataset_name.upper()}"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_path = root / "convergence.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print("Saved:", output_path)
