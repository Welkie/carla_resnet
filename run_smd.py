import os
import sys
import time
import json
import subprocess
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
def run_experiments(data_files, python_exec, phase=0):
    print("\n" + "="*30)
    print(f"STARTING EXPERIMENTS SMD - PHASE {phase}")
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

    for fname in data_files:
        print(f"\nRunning dataset: {fname}")
        start = time.time()

        # Run pretext
        try:
            result_pretext = subprocess.run([
                python_exec, "carla_pretext.py",
                "--config_env", "configs/env.yml",
                "--config_exp", "configs/pretext/carla_pretext_smd.yml",
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
                "--config_exp", "configs/classification/carla_classification_smd.yml",
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

    total_time = sum(execution_times) 
    
    # If running Phase 0 (all at once), use wall clock for total time
    if phase == 0:
        total_time = time.time() - start_all
    
    avg_time = total_time / len(execution_times) if execution_times else 0

    print("\n" + "="*30)
    print(f"DONE PHASE {phase} ({len(data_files)} datasets)")
    print(f"Total time (this phase): {total_time:.2f} s")
    print(f"Avg / dataset: {avg_time:.2f} s")
    print("="*30)

    return {
        "TOTAL_TIME": total_time,
        "AVG_TIME": avg_time,
        "MAX_GPU_MEM_MB": max_gpu_mem_mb,
        "DATASET_COUNT": len(execution_times)
    }

# =========================================================
# EVALUATION (PAPER-STYLE)
# =========================================================
def evaluate_experiments(data_files, prev_metrics_file=None, output_metrics_file=None):
    print("\n" + "="*30)
    print("STARTING EVALUATION SMD")
    print("="*30)

    # DataFrame to store metrics
    res_df = pd.DataFrame(columns=[
        "name", "pr",
        "best_tp", "best_tn", "best_fp", "best_fn"
    ])

    # Load previous metrics if provided
    if prev_metrics_file and os.path.exists(prev_metrics_file):
        try:
            prev_df = pd.read_csv(prev_metrics_file)
            # Ensure columns match
            if not prev_df.empty and all(col in prev_df.columns for col in res_df.columns):
                # Avoid FutureWarning by checking if res_df is empty
                if res_df.empty:
                    res_df = prev_df
                else:
                    res_df = pd.concat([res_df, prev_df], ignore_index=True)
                print(f"Loaded {len(prev_df)} metrics from {prev_metrics_file}")
            else:
                print(f"Warning: {prev_metrics_file} has incompatible format. Ignoring.")
        except Exception as e:
            print(f"Warning: Could not load previous metrics from {prev_metrics_file}: {e}")

    # Process new files
    for fname in data_files:
        # Check if already in res_df (avoid duplicates if re-running)
        if fname in res_df["name"].values:
            print(f"Skipping {fname} (already evaluated)")
            continue

        test_path = f"results/smd/{fname}/classification/classification_testprobs.csv"
        train_path = f"results/smd/{fname}/classification/classification_trainprobs.csv"

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
            tn, fp, fn, tp = confusion_matrix(df_test["Class"], pred).ravel()

            new_row = pd.DataFrame([{
                "name": fname, 
                "pr": pr_auc, 
                "best_tp": tp, "best_tn": tn, "best_fp": fp, "best_fn": fn
            }])
            res_df = pd.concat([res_df, new_row], ignore_index=True)

            print(f"{fname}: PR-AUC={pr_auc:.4f}, TP={tp}, FP={fp}, FN={fn}")

        except Exception as e:
            print(f"Error {fname}: {e}")

    if res_df.empty:
        print("No results!")
        return None

    # Save metrics to file if requested
    if output_metrics_file:
        try:
            res_df.to_csv(output_metrics_file, index=False)
            print(f"Saved metrics to {output_metrics_file}")
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")

    summary = add_summary_statistics(res_df)

    # Save final json summary
    with open("results/smd/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*30)
    print("FINAL RESULTS SMD")
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
    out = "results/smd/ketqua.txt"

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
    parser = argparse.ArgumentParser(description="Run SMD Experiments")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2], 
                        help="Phase of execution: 0=All, 1=First Half, 2=Second Half")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    # Define Kaggle input path
    kaggle_input_path = "/kaggle/input/datasets/mgusat/smd-onmiad/ServerMachineDataset"
    writable_dataset_path = os.path.join(BASE_DIR, "datasets", "smd")
    
    # Check if Kaggle input exists
    if os.path.exists(kaggle_input_path):
        print(f"Found Kaggle dataset at: {kaggle_input_path}")
        import shutil
        
        # Function to safely copy directories
        def safe_copy_dir(src_name, dst_name):
            src = os.path.join(kaggle_input_path, src_name)
            dst = os.path.join(writable_dataset_path, dst_name)
            if os.path.exists(src):
                if not os.path.exists(dst):
                    print(f"Copying {src} to {dst}...")
                    shutil.copytree(src, dst)
                else:
                    print(f"Directory {dst} already exists. Skipping copy.")
            else:
                 print(f"Warning: Source directory {src} not found.")

        safe_copy_dir("train", "train")
        safe_copy_dir("test", "test")
        safe_copy_dir("test_label", "test_label")
        
    else:
        print("Kaggle input path not found. Using local path if available.")

    # Lấy danh sách file dataset trong folder datasets/smd/train
    train_dir = os.path.join("datasets", "smd", "train")
    if not os.path.exists(train_dir):
        print(f"Error: Directory {train_dir} does not exist.")
        return

    # Sắp xếp để chạy theo thứ tự
    all_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.txt')])
    
    if not all_files:
        print(f"No .txt files found in {train_dir}")
        return

    print(f"Found total {len(all_files)} datasets in smd.")

    # Split datasets based on phase
    if args.phase == 0:
        data_files = all_files
    else:
        mid_point = len(all_files) // 2
        if args.phase == 1:
            data_files = all_files[:mid_point]
            print(f"PHASE 1: Running first {len(data_files)} datasets.")
        else: # phase 2
            data_files = all_files[mid_point:]
            print(f"PHASE 2: Running last {len(data_files)} datasets.")

    current_time_stats = run_experiments(data_files, sys.executable, phase=args.phase)
    
    # Configure evaluation paths
    phase_1_metrics_file = "results/smd/phase_1_metrics_df.csv"
    phase_1_time_file = "results/smd/phase_1_time_stats.json"
    
    if args.phase == 1:
        # Phase 1: Evaluate current files and save metrics + time stats
        print("\nEvaluating Phase 1 results...")
        evaluate_experiments(data_files, output_metrics_file=phase_1_metrics_file)
        
        # Save time stats
        with open(phase_1_time_file, "w") as f:
            json.dump(current_time_stats, f, indent=2)
            
        print(f"Phase 1 completed. Please push '{phase_1_metrics_file}' and '{phase_1_time_file}' to continue in Phase 2.")
        
    elif args.phase == 2:
        # Phase 2: Load Phase 1 metrics (if available) and evaluate current files
        print("\nEvaluating Phase 2 results (merging with Phase 1)...")
        eval_results = evaluate_experiments(data_files, prev_metrics_file=phase_1_metrics_file, output_metrics_file="results/smd/full_metrics.csv") 
        
        # Merge time stats
        final_time_stats = current_time_stats.copy()
        if os.path.exists(phase_1_time_file):
            try:
                with open(phase_1_time_file, "r") as f:
                    phase_1_stats = json.load(f)
                    
                    # Merge logic
                    total_time = phase_1_stats.get("TOTAL_TIME", 0) + current_time_stats.get("TOTAL_TIME", 0)
                    dataset_count = phase_1_stats.get("DATASET_COUNT", 0) + current_time_stats.get("DATASET_COUNT", 0)
                    max_mem = max(phase_1_stats.get("MAX_GPU_MEM_MB", 0), current_time_stats.get("MAX_GPU_MEM_MB", 0))
                    
                    avg_time = total_time / dataset_count if dataset_count > 0 else 0
                    
                    final_time_stats = {
                        "TOTAL_TIME": total_time,
                        "AVG_TIME": avg_time,
                        "MAX_GPU_MEM_MB": max_mem,
                        "DATASET_COUNT": dataset_count
                    }
                    print(f"Merged time stats from Phase 1: Total Time={total_time:.2f}s")
            except Exception as e:
                print(f"Warning: Could not load Phase 1 time stats: {e}")
        
        if eval_results:
            write_summary(final_time_stats, eval_results)
            
    else: # Phase 0
        print("\nVerifying all results for evaluation...")
        # Evaluate all files directly 
        eval_results = evaluate_experiments(all_files, output_metrics_file="results/smd/full_metrics.csv")
        
        if eval_results:
            write_summary(current_time_stats, eval_results)

if __name__ == "__main__":
    main()