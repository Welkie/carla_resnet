import os
import sys
import time
import json
import subprocess
import shutil
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# SEED SETTING
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

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
def run_experiments(base_dir, data_info, python_exec, seed=4, wsz=300):
    set_seed(seed)
    print("\n" + "="*30)
    print(f"STARTING EXPERIMENTS MSL (SEED {seed}, WSZ {wsz})")
    print("="*30)
    
    execution_times = []
    max_gpu_mem_mb = 0.0
    start_all = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, memory tracking disabled")

    for fname in data_info["chan_id"]:
        print(f"\nRunning dataset: {fname} (Seed {seed}, WSZ {wsz})")
        start = time.time()

        # Clean previous checkpoint & output directory to ensure 100% independence
        shutil.rmtree(os.path.join("results", "MSL", fname), ignore_errors=True)
        shutil.rmtree(os.path.join("results", "msl", fname), ignore_errors=True)

        # Run pretext
        try:
            result_pretext = subprocess.run([
                python_exec, "-c",
                f"import sys, torch; sys.argv=['carla_pretext.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/pretext/carla_pretext_msl.yml', '--fname', '{fname}', '--wsz', '{wsz}']; import carla_pretext; carla_pretext.set_seed({seed}); carla_pretext.main(); print(f'Max GPU Memory Used: {{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}} MB') if torch.cuda.is_available() else None"
            ], capture_output=True, text=True, check=True)
            
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
                python_exec, "-c",
                f"import sys, torch; sys.argv=['carla_classification.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/classification/carla_classification_msl.yml', '--fname', '{fname}', '--wsz', '{wsz}']; import carla_classification; carla_classification.set_seed({seed}); carla_classification.main(); print(f'Max GPU Memory Used: {{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}} MB') if torch.cuda.is_available() else None"
            ], capture_output=True, text=True, check=True)

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

        if torch.cuda.is_available():
            current_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            max_gpu_mem_mb = max(max_gpu_mem_mb, current_max_mem)
            torch.cuda.reset_peak_memory_stats()

    total_time = time.time() - start_all
    avg_time = total_time / len(execution_times) if execution_times else 0

    print("\n" + "="*30)
    print(f"DONE ALL MSL DATASETS (SEED {seed}, WSZ {wsz})")
    print(f"Total time: {total_time:.2f} s")
    print(f"Avg / dataset: {avg_time:.2f} s")
    print("="*30)

    os.makedirs("results/msl", exist_ok=True)
    time_results = {
        "TOTAL_TIME": total_time,
        "AVG_TIME": avg_time,
        "MAX_GPU_MEM_MB": max_gpu_mem_mb,
        "SEED": seed,
        "WSZ": wsz
    }
    with open(f"results/msl/time_results_seed{seed}_wsz{wsz}.json", "w") as f:
        json.dump(time_results, f, indent=2)
    
    print(f"\nTime results saved to results/msl/time_results_seed{seed}_wsz{wsz}.json")
    return time_results

# =========================================================
# EVALUATION (PAPER-STYLE)
# =========================================================
def evaluate_experiments(data_info, seed=4, wsz=300):
    print("\n" + "="*30)
    print(f"STARTING EVALUATION (PAPER STYLE - SEED {seed}, WSZ {wsz})")
    print("="*30)

    res_df = pd.DataFrame(columns=[
        "name", "pr",
        "best_tp", "best_tn", "best_fp", "best_fn"
    ])

    for fname in data_info["chan_id"]:
        test_path = f"results/MSL/{fname}/classification/classification_testprobs.csv"
        train_path = f"results/MSL/{fname}/classification/classification_trainprobs.csv"

        if not os.path.exists(test_path) or not os.path.exists(train_path):
            test_path = f"results/msl/{fname}/classification/classification_testprobs.csv"
            train_path = f"results/msl/{fname}/classification/classification_trainprobs.csv"

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
            tn, fp, fn, tp = confusion_matrix(df_test["Class"], pred, labels=[0, 1]).ravel()

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

    with open(f"results/msl/evaluation_results_seed{seed}_wsz{wsz}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*30)
    print(f"FINAL RESULTS (PAPER STYLE - SEED {seed}, WSZ {wsz})")
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
def write_summary(time_results, eval_results, seed=4, wsz=300):
    out = "results/msl/ketqua.txt"

    summary_lines = [
        f"================ SUMMARY (SEED {seed}, WSZ {wsz}) ================",
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

    with open(out, "a") as f:
        f.write(summary_text + "\n\n")

    print(f"\nSummary written to {out}")

# =========================================================
# MAIN
# =========================================================
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    kaggle_input_path = "/kaggle/input/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
    writable_dataset_path = os.path.join(BASE_DIR, "datasets", "MSL")

    os.makedirs(writable_dataset_path, exist_ok=True)

    if os.path.exists(kaggle_input_path):
        print(f"Found Kaggle dataset at: {kaggle_input_path}")

        src_csv = os.path.join(kaggle_input_path, "labeled_anomalies.csv")
        dst_csv = os.path.join(writable_dataset_path, "labeled_anomalies.csv")
        if os.path.exists(src_csv) and not os.path.exists(dst_csv):
            print(f"Copying {src_csv} to {dst_csv}...")
            shutil.copyfile(src_csv, dst_csv)
        
        def safe_copy_dir(src_subpath, dst_name):
            src = os.path.join(kaggle_input_path, src_subpath)
            dst = os.path.join(writable_dataset_path, dst_name)
            if os.path.exists(src):
                if not os.path.exists(dst):
                    print(f"Copying {src} to {dst}...")
                    shutil.copytree(src, dst)
                else:
                    print(f"Directory {dst} already exists. Skipping copy.")
            else:
                 print(f"Warning: Source directory {src} not found.")

        safe_copy_dir(os.path.join("data", "data", "train"), "train")
        safe_copy_dir(os.path.join("data", "data", "test"), "test")
        
    else:
        print("Kaggle input path not found. Using local path if available.")

    csv_path = "datasets/MSL/labeled_anomalies.csv"
    data_info = pd.read_csv(csv_path)
    data_info = data_info[data_info["spacecraft"] == "MSL"]

    # Clear previous ketqua.txt file if it exists
    out_txt = "results/msl/ketqua.txt"
    if os.path.exists(out_txt):
        os.remove(out_txt)

    runs = [
        {"seed": 4, "wsz": 1000},
    ]

    for idx, run_cfg in enumerate(runs, 1):
        s = run_cfg["seed"]
        w = run_cfg["wsz"]
        print("\n" + "="*50)
        print(f"Lần {idx}: seed {s}, wsz={w}")
        print("="*50)

        time_results = run_experiments(BASE_DIR, data_info, sys.executable, seed=s, wsz=w)
        eval_results = evaluate_experiments(data_info, seed=s, wsz=w)

        if time_results and eval_results:
            write_summary(time_results, eval_results, seed=s, wsz=w)

if __name__ == "__main__":
    main()