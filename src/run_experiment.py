import argparse
import csv
import json

from .config import (
    FINAL_EPOCHS,
    FINAL_SEEDS,
    RUN_METRICS_DIR,
    SCREENING_CONFIGS,
    SCREENING_CSV,
    SCREENING_EPOCHS,
    SCREENING_SEEDS,
)


def make_run_id(stage, setting, seed):
    return (
        f"{stage}_{setting.method}"
        f"_a{setting.lr_a:g}_b{setting.lr_b:g}"
        f"_lambda{setting.lambda_ratio}_seed{seed}"
    )


def run_or_resume(
    *,
    setting,
    seed,
    stage,
    epochs,
    tokenizer,
    train_dataset,
    validation_dataset,
    data_collator,
):
    """Reuse completed run outputs; otherwise train and save this run."""
    from .metrics import save_run_outputs, save_run_plot
    from .train import train_one_run

    run_id = make_run_id(stage, setting, seed)
    json_path = RUN_METRICS_DIR / f"{run_id}.json"

    if json_path.exists():
        print(f"Already completed; reusing {run_id}")
        with json_path.open(encoding="utf-8") as file:
            result = json.load(file)

        # Rebuild the summary and plot if a previous session ended early.
        save_run_outputs(result)
        plot_path = save_run_plot(result)
        print(f"Saved plot: {plot_path}")
        return result

    return train_one_run(
        setting=setting,
        seed=seed,
        stage=stage,
        epochs=epochs,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        data_collator=data_collator,
    )


def select_best_settings():
    """Select the best standard LoRA and LoRA+ screening settings."""
    if not SCREENING_CSV.exists():
        raise FileNotFoundError(
            "Screening results are missing. Run the screening stage first."
        )

    with SCREENING_CSV.open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row.get("stage") == "screening"
        ]

    expected_ids = {
        make_run_id("screening", setting, seed)
        for setting in SCREENING_CONFIGS
        for seed in SCREENING_SEEDS
    }
    completed_ids = {row.get("run_id") for row in rows}
    missing_ids = expected_ids - completed_ids

    if missing_ids:
        raise RuntimeError(
            f"Screening is incomplete: {len(missing_ids)} run(s) are missing. "
            "Finish all screening configurations before starting final runs."
        )

    selected = {}

    for method in ("standard_lora", "loraplus"):
        method_rows = [
            row for row in rows
            if row.get("method") == method
        ]

        if not method_rows:
            raise RuntimeError(
                f"No completed screening results found for {method}."
            )

        # Highest validation accuracy wins; lower validation loss breaks ties.
        best = min(
            method_rows,
            key=lambda row: (
                -float(row["validation_accuracy"]),
                float(row["validation_loss"]),
            ),
        )

        from .config import RunSetting

        selected[method] = RunSetting(
            method=best["method"],
            lr_a=float(best["lr_a"]),
            lr_b=float(best["lr_b"]),
            lambda_ratio=int(best["lambda_ratio"]),
        )

    return selected


def run_experiments(stage="all", dry_run=False, allow_cpu=False):
    """Run screening, final experiments, or both."""
    if stage not in ("screening", "final", "all"):
        raise ValueError("stage must be 'screening', 'final', or 'all'.")

    # Handle dry-run before importing PyTorch or model-training dependencies.
    if dry_run:
        print(f"Dry run: stage={stage}")

        if stage in ("screening", "all"):
            total_screening_runs = (
                len(SCREENING_CONFIGS) * len(SCREENING_SEEDS)
            )
            print(
                f"Would run {total_screening_runs} screening run(s): "
                f"{len(SCREENING_CONFIGS)} settings × "
                f"{len(SCREENING_SEEDS)} seed(s)"
            )
            print(
                f"Screening seeds: {SCREENING_SEEDS}; "
                f"epochs: {SCREENING_EPOCHS}"
            )

        if stage in ("final", "all"):
            total_final_runs = 2 * len(FINAL_SEEDS)
            print(
                "Final runs require completed screening and select one "
                "setting for each method."
            )
            print(
                f"Would run up to {total_final_runs} final run(s): "
                f"2 methods × {len(FINAL_SEEDS)} seed(s)"
            )
            print(
                f"Final seeds: {FINAL_SEEDS}; "
                f"epochs: {FINAL_EPOCHS}"
            )

        return

    # These imports are needed only when actually launching experiments.
    import torch
    from .data_utils import load_sst2_data

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "No CUDA GPU is available. Run the full experiments in Colab "
            "with a GPU. Use --dry-run to inspect the plan."
        )

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU because --allow-cpu was specified.")

    tokenizer, train_dataset, validation_dataset, data_collator = (
        load_sst2_data()
    )
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")

    if stage in ("screening", "all"):
        for setting in SCREENING_CONFIGS:
            for seed in SCREENING_SEEDS:
                run_or_resume(
                    setting=setting,
                    seed=seed,
                    stage="screening",
                    epochs=SCREENING_EPOCHS,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    data_collator=data_collator,
                )

    if stage in ("final", "all"):
        selected = select_best_settings()

        for method, setting in selected.items():
            print(
                f"Selected {method}: "
                f"LR A={setting.lr_a}, "
                f"LR B={setting.lr_b}, "
                f"lambda={setting.lambda_ratio}"
            )

            for seed in FINAL_SEEDS:
                run_or_resume(
                    setting=setting,
                    seed=seed,
                    stage="final",
                    epochs=FINAL_EPOCHS,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    data_collator=data_collator,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("screening", "final", "all"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA; full runs are expected to be slow.",
    )
    args = parser.parse_args()

    run_experiments(
        stage=args.stage,
        dry_run=args.dry_run,
        allow_cpu=args.allow_cpu,
    )


if __name__ == "__main__":
    main()