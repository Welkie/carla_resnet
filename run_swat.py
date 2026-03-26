import os
import sys
import time
import json
import subprocess
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
import warnings
import shutil
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Try to import kagglehub
try:
    import kagglehub
except ImportError:
    kagglehub = None

# =========================================================
# PAPER-STYLE SUMMARY
# =========================================================
def add_summary_statistics(res_df):
    sum_tp = res_df["best_tp"].sum()
    sum_tn = res_df["best_tn"].sum()
    sum_fp = res_df["best_fp"].sum()
    sum_fn = res_df["best_fn"].sum()

    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    pr_avg = res_df["pr"].mean()
    pr_std = res_df["pr"].std()

    return {
        "PRECISION": precision,
        "RECALL": recall,
        "F1": f1,
        "AUPR_MEAN": pr_avg,
        "AUPR_STD": pr_std,
        "TP": int(sum_tp),
        "TN": int(sum_tn),
        "FP": int(sum_fp),
        "FN": int(sum_fn),
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
    max_gpu_mem_mb = 0.0
    start_all = time.time()

    # Initialize GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, memory tracking disabled")

    for fname in datasets:
        print(f"\nRunning dataset: {fname}")
        start = time.time()

        # Run pretext
        try:
            result_pretext = subprocess.run([
                python_exec, "carla_pretext.py",
                "--config_env", "configs/env.yml",
                "--config_exp", "configs/pretext/carla_pretext_swat.yml",
                "--fname", fname
            ], capture_output=True, text=True, check=True)
            
            # Parse GPU memory from pretext
            if "Max GPU Memory Used:" in result_pretext.stdout:
                for line in result_pretext.stdout.split('\n'):
                    if "Max GPU Memory Used:" in line:
                        mem_str = line.split(": ")[1].split(" MB")[0]
                        max_gpu_mem_mb = max(max_gpu_mem_mb, float(mem_str))
                        break
        except subprocess.CalledProcessError as e:
            print(f"Error running pretext for {fname}: {e}")
            print(e.stderr)

        # Run classification
        try:
            result_classification = subprocess.run([
                python_exec, "carla_classification.py",
                "--config_env", "configs/env.yml",
                "--config_exp", "configs/classification/carla_classification_swat.yml",
                "--fname", fname
            ], capture_output=True, text=True, check=True)

            # Parse GPU memory from classification
            if "Max GPU Memory Used:" in result_classification.stdout:
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

        # Track max GPU memory usage via torch if available locally
        if torch.cuda.is_available():
            current_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            max_gpu_mem_mb = max(max_gpu_mem_mb, current_max_mem)
            torch.cuda.reset_peak_memory_stats()

    total_time = time.time() - start_all
    avg_time = total_time / len(execution_times) if execution_times else 0

    print("\n" + "="*30)
    print("DONE ALL SWAT DATASETS")
    print(f"Total time: {total_time:.2f} s")
    print(f"Avg / dataset: {avg_time:.2f} s")
    print("="*30)

    # Save time results
    os.makedirs("results/swat", exist_ok=True)
    time_results = {
        "TOTAL_TIME": total_time,
        "AVG_TIME": avg_time,
        "MAX_GPU_MEM_MB": max_gpu_mem_mb
    }
    with open("results/swat/time_results.json", "w") as f:
        json.dump(time_results, f, indent=2)
    
    print(f"\nTime results saved to results/swat/time_results.json")
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
        test_path = f"results/swat/{fname}/classification/classification_testprobs.csv"
        train_path = f"results/swat/{fname}/classification/classification_trainprobs.csv"

        if not os.path.exists(test_path) or not os.path.exists(train_path):
            print(f"Skip {fname} (missing files)")
            continue

        try:
            df_test = pd.read_csv(test_path)
            df_train = pd.read_csv(train_path)

            cl_num = df_test.shape[1] - 1

            df_train["pred"] = df_train.iloc[:, :cl_num].idxmax(axis=1)
            normal_class = df_train["pred"].value_counts().idxmax()

            df_test["Class"] = (df_test["Class"] != 0).astype(int)
            scores = 1 - df_test[normal_class]

            pr_auc = average_precision_score(df_test["Class"], scores)

            p, r, t = precision_recall_curve(df_test["Class"], scores)
            f1s = 2 * p * r / (p + r + 1e-9)
            idx = f1s.argmax()
            thr = t[idx]

            pred = scores >= thr
            cm = confusion_matrix(df_test["Class"], pred)
            if cm.size == 1:
                # If only one class is predicted (e.g., all 0 or all 1)
                # Determine which case it is based on the actual values
                if df_test["Class"].iloc[0] == 0: # All true negatives (or false positives if pred=1)
                     tn, fp, fn, tp = cm[0,0], 0, 0, 0 # assumption if pred=0
                     if pred.iloc[0]: # If predicted 1
                         tn, fp, fn, tp = 0, cm[0,0], 0, 0
                else: # All true positives (or false negatives)
                     tn, fp, fn, tp = 0, 0, cm[0,0], 0 # assumption if pred=0
                     if pred.iloc[0]: 
                         tn, fp, fn, tp = 0, 0, 0, cm[0,0]
            else:
                tn, fp, fn, tp = cm.ravel()

            res_df.loc[len(res_df)] = [
                fname, pr_auc, tp, tn, fp, fn
            ]

            print(f"{fname}: PR-AUC={pr_auc:.4f}, TP={tp}, FP={fp}, FN={fn}")

        except Exception as e:
            print(f"Error {fname}: {e}")

    if res_df.empty:
        print("No results!")
        return None

    summary = add_summary_statistics(res_df)

    with open("results/swat/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*30)
    print("FINAL RESULTS (PAPER STYLE)")
    print("="*30)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return summary

# =========================================================
# WRITE SUMMARY
# =========================================================
def write_summary(time_results, eval_results):
    out = "results/swat/ketqua.txt"

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

    # In ra màn hình
    print("\n" + summary_text)

    # Ghi ra file
    with open(out, "w") as f:
        f.write(summary_text + "\n")

    print(f"\nSummary written to {out}")

# =========================================================
# MAIN
# =========================================================
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    datasets = ["swat"]

    # Define the Kaggle input path provided by the user
    kaggle_input_path = "/kaggle/input/datasets/vishala28/swat-dataset-secure-water-treatment-system"

    writable_dataset_path = os.path.join(BASE_DIR, "datasets", "swat")

    # Ensure writable directory exists
    os.makedirs(writable_dataset_path, exist_ok=True)

    # Check if the Kaggle input path exists - Restore compatibility for Kaggle
    if os.path.exists(kaggle_input_path):
        print(f"Found Kaggle dataset at: {kaggle_input_path}")
        
        # files to look for
        files = os.listdir(kaggle_input_path)
        normal_file = None
        attack_file = None

        # Look for merged.csv first as requested by user
        merged_file = None
        if 'merged.csv' in files:
            merged_file = 'merged.csv'
        else:
             for f in files:
                if 'merged' in f.lower() and f.endswith('.csv'):
                    merged_file = f
                    break
        
        if merged_file:
             src = os.path.join(kaggle_input_path, merged_file)
             print(f"Found merged dataset: {src}")
             # Process merged dataset (split into normal.csv and attack.csv)
             # We do this regardless if they exist to ensure fresh split from merged
             process_merged_dataset(src, writable_dataset_path)
        else:
             # Fallback to looking for separate normal and attack files
             print("merged.csv not found. Looking for separate files...")
             
             # Heuristic to find files
             if 'normal.csv' in files:
                 normal_file = 'normal.csv'
             else:
                 for f in files:
                     if 'normal' in f.lower() and f.endswith('.csv'):
                         normal_file = f
                         break
             
             if 'attack.csv' in files:
                 attack_file = 'attack.csv'
             else:
                 for f in files:
                     if 'attack' in f.lower() and f.endswith('.csv'):
                         attack_file = f
                         break
             
             # Copy to writable path with correct names if they don't exist
             if normal_file:
                 src = os.path.join(kaggle_input_path, normal_file)
                 dst = os.path.join(writable_dataset_path, "normal.csv")
                 if not os.path.exists(dst):
                     print(f"Copying {src} to {dst}...")
                     shutil.copyfile(src, dst)
             else:
                 print("Warning: Could not find normal file in Kaggle input")

             if attack_file:
                 src = os.path.join(kaggle_input_path, attack_file)
                 dst = os.path.join(writable_dataset_path, "attack.csv")
                 if not os.path.exists(dst):
                     print(f"Copying {src} to {dst}...")
                     shutil.copyfile(src, dst)
             else:
                 print("Warning: Could not find attack file in Kaggle input")

        os.environ['swat_DATASET_PATH'] = writable_dataset_path
        print(f"Set swat_DATASET_PATH to {writable_dataset_path}")
    else:
        # Local environment fallback
        print("Kaggle input path not found. using local path if available.")
        
    # Check if files exist (Common check)
    if not os.path.exists(os.path.join(writable_dataset_path, "normal.csv")) or \
       not os.path.exists(os.path.join(writable_dataset_path, "attack.csv")):
        print(f"Warning: normal.csv or attack.csv not found in {writable_dataset_path}")
        print("Please ensure 'datasets/swat/normal.csv' and 'datasets/swat/attack.csv' exist.")
    
    os.environ['swat_DATASET_PATH'] = writable_dataset_path
    print(f"Set swat_DATASET_PATH to {writable_dataset_path}")


    time_results = run_experiments(BASE_DIR, datasets, sys.executable)
    eval_results = evaluate_experiments(datasets)

    if time_results and eval_results:
        write_summary(time_results, eval_results)

def process_merged_dataset(merged_path, output_dir):
    """
    Splits the merged.csv into normal.csv (Train) and attack.csv (Test).
    Standard SWAT split: 
    - Train (Normal): First ~7 days. 
    - Test (Attack): Remaining ~4 days (contains both Normal and Attack).
    
    Standard row counts (approx):
    - Train: 496800
    - Test: 449919
    """
    print(f"Processing merged dataset from {merged_path}...")
    try:
        df = pd.read_csv(merged_path)
        # Strip whitespace from columns
        df.columns = df.columns.str.strip()
        
        # Determine split point
        # Option 1: Hardcoded standard SWAT split
        split_index = 496800
        
        if len(df) < split_index:
             print(f"Warning: Dataset length ({len(df)}) is smaller than standard split ({split_index}). Using 50% split.")
             split_index = len(df) // 2
        
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        
        # Save to destination
        train_path = os.path.join(output_dir, "normal.csv")
        test_path = os.path.join(output_dir, "attack.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Split merged.csv into:")
        print(f"  Train (normal.csv): {len(train_df)} rows")
        print(f"  Test (attack.csv): {len(test_df)} rows")
        
    except Exception as e:
        print(f"Error processing merged dataset: {e}")
        raise e

if __name__ == "__main__":
    main()