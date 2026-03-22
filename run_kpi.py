import os
import sys
import time
import json
import subprocess
import shutil
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
# PREPARE KPI DATA — HDF5 → per-KPI train/test CSVs
# =========================================================
def prepare_kpi_data(hdf_path, writable_dataset_path):
    """
    Reads the phase2_ground_truth.hdf file which contains all 29 KPIs.
    Splits each KPI 50/50 (first half = train, second half = test).
    Saves per-KPI CSVs to:
        datasets/kpi/train/<kpi_name>.csv
        datasets/kpi/test/<kpi_name>.csv
    Columns: timestamp, value, label
    Train label column is kept (all values preserved) but CARLA trains unsupervised.
    Returns the sorted list of CSV filenames (e.g. ["kpi_0.csv", "kpi_1.csv", ...]).
    """
    train_dir = os.path.join(writable_dataset_path, "train")
    test_dir  = os.path.join(writable_dataset_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    print(f"Reading HDF5 file: {hdf_path}")
    # The HDF5 stores a DataFrame; try reading as HDF store first, then fallback to direct read
    try:
        df_all = pd.read_hdf(hdf_path, key='data')
    except Exception:
        # Some HDF files use a default key or different structure
        try:
            store = pd.HDFStore(hdf_path, 'r')
            key = store.keys()[0]
            df_all = store[key]
            store.close()
        except Exception as e:
            print(f"ERROR reading HDF5: {e}")
            raise

    print(f"HDF5 loaded — shape: {df_all.shape}, columns: {list(df_all.columns)}")

    # === Normalise column names (case-insensitive) ===
    col_lower = {c.lower(): c for c in df_all.columns}

    # Identify KPI-id / group column
    kpi_col = None
    for cand in ['kpi_id', 'kpi id', 'series', 'metric', 'name', 'id']:
        if cand in col_lower:
            kpi_col = col_lower[cand]
            break
    if kpi_col is None:
        # last-resort: pick the non-numeric column that is not timestamp/value/label
        for c in df_all.columns:
            if df_all[c].dtype == object:
                kpi_col = c
                break

    value_col     = col_lower.get('value',     col_lower.get('values', None))
    label_col     = col_lower.get('label',     col_lower.get('labels', col_lower.get('anomaly', None)))
    timestamp_col = col_lower.get('timestamp', col_lower.get('time',   None))

    print(f"Detected columns — kpi_id: {kpi_col}, value: {value_col}, "
          f"label: {label_col}, timestamp: {timestamp_col}")

    if kpi_col is None or value_col is None or label_col is None:
        raise ValueError(
            f"Cannot identify required columns. Available: {list(df_all.columns)}"
        )

    file_list = []
    kpi_groups = sorted(df_all[kpi_col].unique())
    print(f"Found {len(kpi_groups)} KPI series.")

    for kpi_id in kpi_groups:
        kpi_df = df_all[df_all[kpi_col] == kpi_id].copy()

        # Sort by timestamp if available
        if timestamp_col is not None:
            kpi_df = kpi_df.sort_values(by=timestamp_col)

        # Build a clean DataFrame with exactly the columns KPI.py expects
        if timestamp_col is not None:
            out_df = pd.DataFrame({
                'timestamp': kpi_df[timestamp_col].values,
                'value':     kpi_df[value_col].values,
                'label':     kpi_df[label_col].values.astype(int)
            })
        else:
            out_df = pd.DataFrame({
                'timestamp': np.arange(len(kpi_df)),
                'value':     kpi_df[value_col].values,
                'label':     kpi_df[label_col].values.astype(int)
            })

        # 50/50 split
        split_idx = len(out_df) // 2
        train_df  = out_df.iloc[:split_idx].reset_index(drop=True)
        test_df   = out_df.iloc[split_idx:].reset_index(drop=True)

        # Sanitise KPI name for use as filename
        safe_name = str(kpi_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
        csv_name  = f"{safe_name}.csv"

        train_path = os.path.join(train_dir, csv_name)
        test_path  = os.path.join(test_dir,  csv_name)

        # Only write if not already present (idempotent re-runs)
        if not os.path.exists(train_path):
            train_df.to_csv(train_path, index=False)
        if not os.path.exists(test_path):
            test_df.to_csv(test_path, index=False)

        file_list.append(csv_name)

        n_anom_train = int(train_df['label'].sum())
        n_anom_test  = int(test_df['label'].sum())
        print(f"  KPI {safe_name}: train={len(train_df)} rows ({n_anom_train} anomalies), "
              f"test={len(test_df)} rows ({n_anom_test} anomalies)")

    file_list = sorted(file_list)
    print(f"\nPrepared {len(file_list)} KPI CSVs → datasets/kpi/train/ and datasets/kpi/test/")
    return file_list


# =========================================================
# RUN EXPERIMENTS
# =========================================================
def run_experiments(base_dir, file_list, python_exec, phase=0):
    print("\n" + "="*30)
    print(f"STARTING EXPERIMENTS KPI - PHASE {phase}")
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

    for fname in file_list:
        print(f"\nRunning dataset: {fname}")
        start = time.time()

        # Run pretext
        try:
            result_pretext = subprocess.run([
                python_exec, "carla_pretext.py",
                "--config_env", "configs/env.yml",
                "--config_exp", "configs/pretext/carla_pretext_kpi.yml",
                "--fname", fname
            ], capture_output=True, text=True, check=True)

            # Parse GPU memory from pretext output
            if "Max GPU Memory Used:" in result_pretext.stdout:
                for line in result_pretext.stdout.split('\n'):
                    if "Max GPU Memory Used:" in line:
                        mem_str = line.split(": ")[1].split(" MB")[0]
                        max_gpu_mem_mb = max(max_gpu_mem_mb, float(mem_str))
                        break
        except subprocess.CalledProcessError as e:
            print(f"Error running pretext for {fname}: {e}")
            print(e.stderr[-3000:])  # Print last 3000 chars of stderr

        # Run classification
        try:
            result_classification = subprocess.run([
                python_exec, "carla_classification.py",
                "--config_env", "configs/env.yml",
                "--config_exp", "configs/classification/carla_classification_kpi.yml",
                "--fname", fname
            ], capture_output=True, text=True, check=True)

            # Parse GPU memory from classification output
            if "Max GPU Memory Used:" in result_classification.stdout:
                for line in result_classification.stdout.split('\n'):
                    if "Max GPU Memory Used:" in line:
                        mem_str = line.split(": ")[1].split(" MB")[0]
                        max_gpu_mem_mb = max(max_gpu_mem_mb, float(mem_str))
                        break
        except subprocess.CalledProcessError as e:
            print(f"Error running classification for {fname}: {e}")
            print(e.stderr[-3000:])

        execution_times.append(time.time() - start)
        print(f"Max GPU Memory after {fname}: {max_gpu_mem_mb:.2f} MB")

        # Also track via torch directly if available
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
    print(f"DONE PHASE {phase} ({len(file_list)} datasets)")
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
def evaluate_experiments(file_list, prev_metrics_file=None, output_metrics_file=None):
    print("\n" + "="*30)
    print("STARTING EVALUATION KPI")
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
    for fname in file_list:
        # Check if already in res_df (avoid duplicates if re-running)
        if fname in res_df["name"].values:
            print(f"Skipping {fname} (already evaluated)")
            continue

        test_path  = f"results/kpi/{fname}/classification/classification_testprobs.csv"
        train_path = f"results/kpi/{fname}/classification/classification_trainprobs.csv"

        if not os.path.exists(test_path) or not os.path.exists(train_path):
            print(f"Skip {fname} (missing files)")
            continue

        try:
            df_test  = pd.read_csv(test_path)
            df_train = pd.read_csv(train_path)

            cl_num = df_test.shape[1] - 1  # number of class columns

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
    os.makedirs("results/kpi", exist_ok=True)
    with open("results/kpi/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*30)
    print("FINAL RESULTS KPI")
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
    out = "results/kpi/ketqua.txt"

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

    # Print to stdout
    print("\n" + summary_text)

    # Write to file
    with open(out, "w") as f:
        f.write(summary_text + "\n")

    print(f"\nSummary written to {out}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Run KPI Experiments")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2], 
                        help="Phase of execution: 0=All, 1=First Half, 2=Second Half")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    # ===========================================================
    # KPI-Anomaly-Detection dataset on Kaggle
    # HDF5 path: /kaggle/input/datasets/minhanhphm1676/kpi-anomaly-dectection/
    #            KPI-Anomaly-Detection-master/Finals_dataset/
    #            phase2_ground_truth.hdf/phase2_ground_truth.hdf
    # 29 KPI series, ~2.9 M rows total.
    # Split per KPI: first 50% → train (unsupervised), second 50% → test.
    # ===========================================================
    kaggle_hdf_path = (
        "/kaggle/input/datasets/minhanhphm1676/kpi-anomaly-dectection"
        "/KPI-Anomaly-Detection-master/Finals_dataset"
        "/phase2_ground_truth.hdf/phase2_ground_truth.hdf"
    )
    writable_dataset_path = os.path.join(BASE_DIR, "datasets", "kpi")

    # Ensure writable directory exists
    os.makedirs(writable_dataset_path, exist_ok=True)

    train_dir = os.path.join(writable_dataset_path, "train")
    test_dir  = os.path.join(writable_dataset_path, "test")

    # Prepare data only if not already extracted
    existing_train_files = (
        [f for f in os.listdir(train_dir) if f.endswith(".csv")]
        if os.path.exists(train_dir) else []
    )

    if existing_train_files:
        print(f"Found {len(existing_train_files)} existing KPI CSVs in datasets/kpi/train/. "
              "Skipping extraction.")
        file_list = sorted(existing_train_files)
    elif os.path.exists(kaggle_hdf_path):
        print(f"Found Kaggle HDF5 at: {kaggle_hdf_path}")
        file_list = prepare_kpi_data(kaggle_hdf_path, writable_dataset_path)
    else:
        # Fallback: look for local HDF5 in datasets/kpi/
        local_hdf = os.path.join(writable_dataset_path, "phase2_ground_truth.hdf")
        if os.path.exists(local_hdf):
            print(f"Using local HDF5 at: {local_hdf}")
            file_list = prepare_kpi_data(local_hdf, writable_dataset_path)
        else:
            print("ERROR: HDF5 file not found at Kaggle path or local fallback.")
            print(f"  Expected Kaggle path : {kaggle_hdf_path}")
            print(f"  Expected local path  : {local_hdf}")
            print("Please place the HDF5 file in datasets/kpi/ or ensure the Kaggle dataset is attached.")
            return

    if not file_list:
        print("ERROR: No KPI CSV files found or generated. Exiting.")
        return

    # Split datasets based on phase
    if args.phase == 0:
        data_files = file_list
    else:
        mid_point = len(file_list) // 2
        if args.phase == 1:
            data_files = file_list[:mid_point]
            print(f"PHASE 1: Running first {len(data_files)} datasets.")
        else: # phase 2
            data_files = file_list[mid_point:]
            print(f"PHASE 2: Running last {len(data_files)} datasets.")

    current_time_stats = run_experiments(BASE_DIR, data_files, sys.executable, phase=args.phase)

    # Configure evaluation paths
    phase_1_metrics_file = "results/kpi/phase_1_metrics_df.csv"
    phase_1_time_file = "results/kpi/phase_1_time_stats.json"

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
        eval_results = evaluate_experiments(data_files, prev_metrics_file=phase_1_metrics_file, output_metrics_file="results/kpi/full_metrics.csv") 
        
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
        eval_results = evaluate_experiments(file_list, output_metrics_file="results/kpi/full_metrics.csv")
        
        if eval_results:
            write_summary(current_time_stats, eval_results)


if __name__ == "__main__":
    main()
