#!/usr/bin/env python3
import argparse
import ast
import re
from pathlib import Path

import pandas as pd


RESULT_RE = re.compile(
    r"Setting:\s*(?P<setting>[^,]+),\s*"
    r"RMSE:\s*(?P<rmse>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
    r"MAE:\s*(?P<mae>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
    r"MAPE:\s*(?P<mape>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
    r"SMAPE:\s*(?P<smape>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def get_dataset_tag(config_path: Path) -> str:
    text = config_path.read_text()

    try:
        import yaml

        config = yaml.safe_load(text)
        dataset_map = config.get("data", {}).get("dataset_map", [])
    except Exception:
        match = re.search(r"^\s*dataset_map:\s*(.+?)\s*$", text, re.MULTILINE)
        dataset_map = ast.literal_eval(match.group(1)) if match else []

    if isinstance(dataset_map, str):
        dataset_map = [dataset_map]

    tag = "+".join(str(item) for item in dataset_map) if dataset_map else config_path.stem
    tag = re.sub(r"[^A-Za-z0-9._+-]+", "_", tag).strip("_")
    return tag or config_path.stem


def parse_logs(log_dir: Path, pattern: str) -> pd.DataFrame:
    rows = []

    for log_path in sorted(log_dir.glob(pattern)):
        last_row = None

        for line in log_path.read_text(errors="replace").splitlines():
            match = RESULT_RE.search(line)
            if not match:
                continue

            last_row = {
                "log_file": log_path.name,
                "setting": match.group("setting").strip(),
                "RMSE": float(match.group("rmse")),
                "MAE": float(match.group("mae")),
                "MAPE": float(match.group("mape")),
                "SMAPE": float(match.group("smape")),
            }

        if last_row is not None:
            rows.append(last_row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize UniTime metrics across seeds.")
    parser.add_argument("--config_filename", default="data_configs/roma.yaml")
    # parser.add_argument("--exp_name", default="svd_vot_tr4_se8")
    parser.add_argument("--exp_name", default="baseline")
    # parser.add_argument("--exp_name", default="wo_svd_baseline")
    # parser.add_argument("--exp_name", default="wo_vot_baseline")
    # parser.add_argument("--exp_name", default="wo_confu_baseline")
    # parser.add_argument("--exp_name", default="without_selector_baseline")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--pattern", default="")
    parser.add_argument("--out_csv", default="")
    args = parser.parse_args()

    dataset_tag = get_dataset_tag(Path(args.config_filename))
    run_name = f"{dataset_tag}_{args.exp_name}"
    pattern = args.pattern or f"{run_name}_seed*.log"
    out_csv = args.out_csv or f"{run_name}_summary.csv"

    df = parse_logs(Path(args.log_dir), pattern)
    if df.empty:
        raise SystemExit(f"No result lines found in {args.log_dir}/{pattern}")

    metrics = ["RMSE", "MAE", "MAPE", "SMAPE"]
    summary = df.groupby("setting")[metrics].agg(["mean", "std", "count"])

    flat_summary = summary.copy()
    flat_summary.columns = [f"{metric}_{stat}" for metric, stat in flat_summary.columns]
    flat_summary = flat_summary.reset_index()

    for _, row in flat_summary.iterrows():
        print(f"Setting: {row['setting']}  n={int(row['RMSE_count'])}")
        for metric in metrics:
            print(
                f"  {metric}: "
                f"{row[f'{metric}_mean']:.6f} +- {row[f'{metric}_std']:.6f}"
            )

    flat_summary.to_csv("./result/" + out_csv, index=False)
    print(f"\nSaved CSV: ./result/{out_csv}")


if __name__ == "__main__":
    main()
