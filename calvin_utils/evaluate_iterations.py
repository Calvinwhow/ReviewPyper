#!/usr/bin/env python3
"""
Evaluate GPT predictions vs ground truth per iteration and track accuracy (class-based).

Flags:
  --groundTruth         Path to groundTruth.csv (must contain MRN + question columns)
  --masterList          Path to gptMasterList.csv (must contain MRN + same question columns; 1.0 true, 0.0 false, 2.0 maybe)
  --iterationHistory    Path to aucIterationCalc.csv (append/create)
  --accuracyImage       Path to output accuracy plot (png)
  --reset               Reset all previous iterations and start from 1 (truncates iterationHistory)

Logic:
- INNER JOIN on MRN (compare only overlapping subjects).
- Compare only ground truth question columns (assumed present in masterList).
- MasterList value 2.0 (“maybe”) is excluded from TP/TN/FP/FN.
- Accuracy (“AUC” as defined) = (TP + TN) / (TP + TN + FP + FN).
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class IterationEvaluator:
    """Encapsulates reading inputs, comparing, logging, and plotting."""

    def __init__(self, ground_path: str, master_path: str,
                 history_path: str, plot_path: str, reset: bool = False):
        self.ground_path = ground_path
        self.master_path = master_path
        self.history_path = history_path
        self.plot_path = plot_path
        self.reset = reset

        self.gt = None        # ground truth DataFrame
        self.gpt = None       # master list DataFrame
        self.merged = None    # merged comparison DataFrame
        self.gt_q_cols = []   # ground-truth question columns

    # ----- Normalizers (same behavior as your script) -----
    @staticmethod
    def normalize_truth(x):
        """Normalize ground-truth values to {0,1}; return None if not interpretable."""
        if pd.isna(x):
            return None
        if isinstance(x, (int, np.integer)):
            return 1 if x != 0 else 0
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return None
            return 1 if x != 0.0 else 0
        s = str(x).strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return 1
        if s in {"0", "false", "f", "no", "n"}:
            return 0
        try:
            val = float(s)
            return 1 if val != 0.0 else 0
        except ValueError:
            return None

    @staticmethod
    def normalize_pred_masterlist(x):
        """
        Normalize GPT master list predictions:
          - 1.0 -> 1 (True)
          - 0.0 -> 0 (False)
          - 2.0 -> None (Maybe; exclude)
        Any other value -> None.
        """
        if pd.isna(x):
            return None
        try:
            val = float(x)
        except Exception:
            return None
        if val == 1.0:
            return 1
        if val == 0.0:
            return 0
        if val == 2.0:
            return None
        return None

    # ----- Core steps -----
    def load_inputs(self):
        self.gt = pd.read_csv(self.ground_path)
        self.gpt = pd.read_csv(self.master_path)
        # self.gt.rename(columns={col:col.replace('"','').replace("'",'') for col in self.gt.columns}, inplace=True)
        # self.gpt.rename(columns={col:col.replace('"','').replace("'",'') for col in self.gpt.columns}, inplace=True)

        if "MRN" not in self.gt.columns:
            # print(self.gt.columns[0],len(self.gt.columns))
            raise ValueError("groundTruth.csv must contain an 'MRN' column.")
        if "MRN" not in self.gpt.columns:
            # print(self.gpt.columns[0],len(self.gpt.columns))
            raise ValueError("gptMasterList.csv must contain an 'MRN' column.")

        self.gt_q_cols = [c for c in self.gt.columns if c != "MRN"]
        missing_in_gpt = [c for c in self.gt_q_cols if c not in self.gpt.columns]
        if missing_in_gpt:
            raise ValueError(f"The following ground-truth columns are missing in masterList: {missing_in_gpt}")

        # Inner join on MRN (overlapping subjects only)
        self.gt["MRN"] = self.gt["MRN"].astype(str)
        self.gpt["MRN"] = self.gpt["MRN"].astype(str)
        self.merged = pd.merge(
            self.gt[["MRN"] + self.gt_q_cols],
            self.gpt[["MRN"] + self.gt_q_cols],
            on="MRN",
            how="inner",
            suffixes=("__truth", "__pred"),
        )
        if self.merged.empty:
            raise ValueError("No overlapping MRNs between ground truth and master list.")

    def compute_confusion(self):
        """Compute TP, TN, FP, FN over all subject–question pairs with valid GPT values (0/1)."""
        tp = tn = fp = fn = 0
        for col in self.gt_q_cols:
            tvals = self.merged[f"{col}__truth"].apply(self.normalize_truth)
            pvals = self.merged[f"{col}__pred"].apply(self.normalize_pred_masterlist)
            for t, p in zip(tvals, pvals):
                if t is None or p is None:
                    continue
                if t == 1 and p == 1:
                    tp += 1
                elif t == 0 and p == 0:
                    tn += 1
                elif t == 0 and p == 1:
                    fp += 1
                elif t == 1 and p == 0:
                    fn += 1
        return tp, tn, fp, fn

    @staticmethod
    def next_iteration(log_path: str, reset: bool = False) -> int:
        """Return next iteration number; if reset, return 1 (caller truncates)."""
        if reset or not os.path.exists(log_path):
            return 1
        try:
            df = pd.read_csv(log_path)
            if "iteration" in df.columns and not df.empty:
                return int(df["iteration"].max()) + 1
        except Exception:
            pass
        return 1

    def append_history_and_plot(self, row: dict):
        """Append the row to history CSV and (re)plot accuracy vs iteration."""
        # Reset behavior
        if self.reset and os.path.exists(self.history_path):
            os.remove(self.history_path)

        if os.path.exists(self.history_path):
            log_df = pd.read_csv(self.history_path)
            log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
        else:
            log_df = pd.DataFrame([row])

        log_df.to_csv(self.history_path, index=False)
        acc = row["accuracy"]
        if np.isnan(acc):
            print(f"Iteration {row['iteration']}: accuracy=NaN (no valid pairs; masterList '2.0' may exclude all)")
        else:
            print(f"Iteration {row['iteration']}: accuracy={acc:.4f} (tp={row['tp']}, tn={row['tn']}, fp={row['fp']}, fn={row['fn']})")
        print(f"Updated log: {self.history_path}")

        # Plot
        try:
            plt.figure(figsize=(7, 4.5))
            plt.plot(log_df["iteration"], log_df["accuracy"], marker="o")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.title("Accuracy by Iteration")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=150)
            plt.close()
            print(f"Updated plot: {self.plot_path}")
        except Exception as e:
            print(f"Could not update plot: {e}")

    # ----- Public entry point -----
    def run(self):
        """Execute one evaluation iteration: load, merge, compute, log, and plot."""
        self.load_inputs()
        tp, tn, fp, fn = self.compute_confusion()
        denom = tp + tn + fp + fn
        accuracy = (tp + tn) / denom if denom > 0 else np.nan

        iteration = self.next_iteration(self.history_path, reset=self.reset)

        row = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_subjects": self.merged["MRN"].nunique(),
            "n_labels_compared": len(self.gt_q_cols),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy,
        }

        self.append_history_and_plot(row)
        return row  # optionally return the just-added record


# ---------- CLI ----------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute accuracy vs ground truth and append to iteration log.")
    parser.add_argument("--groundTruth", required=True, help="Path to groundTruth.csv (must contain MRN).")
    parser.add_argument("--masterList", required=True, help="Path to gptMasterList.csv (MRN + questions; 1.0/0.0/2.0).")
    parser.add_argument("--iterationHistory", required=True, help="Path to aucIterationCalc.csv (append/create).")
    parser.add_argument("--accuracyImage", required=True, help="Path to output accuracy plot (png).")
    parser.add_argument("--reset", action="store_true", help="Reset all previous iterations and start from 1.")
    args = parser.parse_args(argv)

    evaluator = IterationEvaluator(
        ground_path=args.groundTruth,
        master_path=args.masterList,
        history_path=args.iterationHistory,
        plot_path=args.accuracyImage,
        reset=args.reset,
    )
    evaluator.run()


if __name__ == "__main__":
    main()

