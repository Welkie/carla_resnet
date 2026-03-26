import os
import sys
import time
import json
import shutil
import subprocess
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# KAGGLE DATASET PATHS
# =========================================================
KAGGLE_TRAIN_PATH = "/kaggle/input/datasets/giovannimonco/wadi-data/WADI_14days_new.csv"
KAGGLE_TEST_PATH  = "/kaggle/input/datasets/giovannimonco/wadi-data/WADI_attackdataLABLE.csv"

# =========================================================
# PAPER-STYLE SUMMARY
# =========================================================
def add_summary_statistics(res_df):
    sum_tp = res_df["best_tp"].sum()
    sum_tn = res_df["best_tn"].sum()
    sum_fp = res_df["best_fp"].sum()
    sum_fn = res_df["best_fn"].sum()

    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall    = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    pr_avg = res_df["pr"].mean()
    pr_std = res_df["pr"].std()

    return {
        "PRECISION":      precision,
        "RECALL":         recall,
        "F1":             f1,
        "AUPR_MEAN":      pr_avg,
        "AUPR_STD":       pr_std,
        "TP":             int(sum_tp),
        "TN":             int(sum_tn),
        "FP":             int(sum_fp),
        "FN":             int(sum_fn),
        "TOTAL_DATASETS": len(res_df)
    }

# =========================================================
# RUN EXPERIMENTS
# =========================================================
def run_experiments(base_dir, datasets, python_exec):
    print("\n" + "="*30)
    print("STARTING EXPERIMENTS")
    print("="*30)

    execution_times = []
    max_gpu_mem_mb  = 0.0
    start_all       = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, memory tracking disabled")

    for fname in datasets:
        print(f"\nRunning dataset: {fname}")
        start = time.time()

        # -- Pretext --
        try:
            result_pretext = subprocess.run([
                python_exec, "carla_pretext.py",
                "--config_env", "configs/env.yml",
                "--config_exp", "configs/pretext/carla_pretext_wadi.yml",
                "--fname", fname
            ], capture_output=True, text=True, check=True)

            for line in result_pretext.stdout.split('\n'):
                if "Max GPU Memory Used:" in line:
                    mem_str = line.split(": ")[1].split(" MB")[0]
                    max_gpu_mem_mb = max(max_gpu_mem_mb, float(mem_str))
                    break
        except subprocess.CalledProcessError as e:
            print(f"Error running pretext for {fname}: {e}")
            print(e.stderr)

        # -- Classification --
        try:
            result_classification = subprocess.run([
                python_exec, "carla_classification.py",
                "--config_env", "configs/env.yml",
                "--config_exp", "configs/classification/carla_classification_wadi.yml",
                "--fname", fname
            ], capture_output=True, text=True, check=True)

            for line in result_classification.stdout.split('\n'):
                if "Max GPU Memory Used:" in line:
                    mem_str = line.split(": ")[1].split(" MB")[0]
                    max_gpu_mem_mb = max(max_gpu_mem_mb, float(mem_str))
                    break
        except subprocess.CalledProcessError as e:
            print(f"Error running classification for {fname}: {e}")
            print(e.stderr)

        execution_times.append(time.time() - start)
        print(f"Max GPU Memory after {fname}: {max_gpu_mem_mb:.2f} MB")

        if torch.cuda.is_available():
            current_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            max_gpu_mem_mb  = max(max_gpu_mem_mb, current_max_mem)
            torch.cuda.reset_peak_memory_stats()

    total_time = time.time() - start_all
    avg_time   = total_time / len(execution_times) if execution_times else 0

    print("\n" + "="*30)
    print("DONE ALL WADI DATASETS")
    print(f"Total time: {total_time:.2f} s")
    print(f"Avg / dataset: {avg_time:.2f} s")
    print("="*30)

    os.makedirs("results/wadi", exist_ok=True)
    time_results = {
        "TOTAL_TIME":     total_time,
        "AVG_TIME":       avg_time,
        "MAX_GPU_MEM_MB": max_gpu_mem_mb
    }
    with open("results/wadi/time_results.json", "w") as f:
        json.dump(time_results, f, indent=2)

    print(f"\nTime results saved to results/wadi/time_results.json")
    return time_results

# =========================================================
# EVALUATION (PAPER-STYLE)
# =========================================================
def evaluate_experiments(datasets):
    print("\n" + "="*30)
    print("STARTING EVALUATION (PAPER STYLE)")
    print("="*30)

    res_df = pd.DataFrame(columns=[
        "name", "pr",
        "best_tp", "best_tn", "best_fp", "best_fn"
    ])

    for fname in datasets:
        test_path  = f"results/wadi/{fname}/classification/classification_testprobs.csv"
        train_path = f"results/wadi/{fname}/classification/classification_trainprobs.csv"

        if not os.path.exists(test_path) or not os.path.exists(train_path):
            print(f"Skip {fname} (missing files)")
            continue

        try:
            df_test  = pd.read_csv(test_path)
            df_train = pd.read_csv(train_path)

            cl_num = df_test.shape[1] - 1

            df_train["pred"] = df_train.iloc[:, :cl_num].idxmax(axis=1)
            normal_class     = df_train["pred"].value_counts().idxmax()

            df_test["Class"] = (df_test["Class"] != 0).astype(int)
            scores = 1 - df_test[normal_class]

            pr_auc = average_precision_score(df_test["Class"], scores)

            p, r, t = precision_recall_curve(df_test["Class"], scores)
            f1s     = 2 * p * r / (p + r + 1e-9)
            idx     = f1s.argmax()
            thr     = t[idx]

            pred = scores >= thr
            cm   = confusion_matrix(df_test["Class"], pred)
            if cm.size == 1:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = cm.ravel()

            res_df.loc[len(res_df)] = [fname, pr_auc, tp, tn, fp, fn]
            print(f"{fname}: PR-AUC={pr_auc:.4f}, TP={tp}, FP={fp}, FN={fn}")

        except Exception as e:
            print(f"Error {fname}: {e}")

    if res_df.empty:
        print("No results!")
        return None

    summary = add_summary_statistics(res_df)

    with open("results/wadi/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*30)
    print("FINAL RESULTS (PAPER STYLE)")
    print("="*30)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    return summary

# =========================================================
# WRITE SUMMARY
# =========================================================
def write_summary(time_results, eval_results):
    out = "results/wadi/ketqua.txt"

    summary_lines = [
        "================ SUMMARY ================",
        f"Precision : {eval_results['PRECISION']:.4f}",
        f"Recall    : {eval_results['RECALL']:.4f}",
        f"F1-score  : {eval_results['F1']:.4f}",
        f"AUPR mean : {eval_results['AUPR_MEAN']:.4f}",
        f"AUPR std  : {eval_results['AUPR_STD']:.4f}",
        "",
        f"Total time     : {time_results['TOTAL_TIME']:.2f} s",
        f"Avg / dataset  : {time_results['AVG_TIME']:.2f} s",
        f"GPU max memory : {time_results['MAX_GPU_MEM_MB']:.2f} MB",
        "========================================="
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(out, "w") as f:
        f.write(summary_text + "\n")

    print(f"\nSummary written to {out}")

# =========================================================
# MAIN
# =========================================================
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    datasets = ["wadi"]

    writable_dataset_path = os.path.join(BASE_DIR, "datasets", "wadi")
    os.makedirs(writable_dataset_path, exist_ok=True)

    # -------------------------------------------------------
    # Copy Kaggle files to writable path (Kaggle input is read-only)
    # -------------------------------------------------------
    kaggle_train = KAGGLE_TRAIN_PATH
    kaggle_test  = KAGGLE_TEST_PATH

    if os.path.exists(kaggle_train):
        dst_train = os.path.join(writable_dataset_path, "WADI_14days_new.csv")
        if not os.path.exists(dst_train):
            print(f"Copying train file: {kaggle_train} → {dst_train}")
            shutil.copyfile(kaggle_train, dst_train)
        else:
            print(f"Train file already exists: {dst_train}")
    else:
        print(f"[INFO] Kaggle train path not found: {kaggle_train}")
        print("       Expecting datasets/wadi/WADI_14days_new.csv to exist locally.")

    if os.path.exists(kaggle_test):
        dst_test = os.path.join(writable_dataset_path, "WADI_attackdataLABLE.csv")
        if not os.path.exists(dst_test):
            print(f"Copying test file:  {kaggle_test} → {dst_test}")
            shutil.copyfile(kaggle_test, dst_test)
        else:
            print(f"Test file already exists: {dst_test}")
    else:
        print(f"[INFO] Kaggle test path not found: {kaggle_test}")
        print("       Expecting datasets/wadi/WADI_attackdataLABLE.csv to exist locally.")

    # -------------------------------------------------------
    # Verify files exist before running
    # -------------------------------------------------------
    train_ok = os.path.exists(os.path.join(writable_dataset_path, "WADI_14days_new.csv"))
    test_ok  = os.path.exists(os.path.join(writable_dataset_path, "WADI_attackdataLABLE.csv"))

    if not train_ok or not test_ok:
        print("\n[ERROR] Dataset files not found. Please check paths:")
        print(f"  Train : {os.path.join(writable_dataset_path, 'WADI_14days_new.csv')} — {'OK' if train_ok else 'MISSING'}")
        print(f"  Test  : {os.path.join(writable_dataset_path, 'WADI_attackdataLABLE.csv')} — {'OK' if test_ok else 'MISSING'}")
        sys.exit(1)

    # Expose path to data/WADI.py via environment variable
    os.environ['wadi_DATASET_PATH'] = writable_dataset_path
    print(f"Set wadi_DATASET_PATH to {writable_dataset_path}")

    # -------------------------------------------------------
    # Run
    # -------------------------------------------------------
    time_results = run_experiments(BASE_DIR, datasets, sys.executable)
    eval_results = evaluate_experiments(datasets)

    if time_results and eval_results:
        write_summary(time_results, eval_results)


if __name__ == "__main__":
    main()
