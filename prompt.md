sửa run_swat.py theo code và hướng dẫn gợi ý dưới đây:

def process_merged_dataset(merged_path, output_dir):
    df = pd.read_csv(merged_path)
    df.columns = df.columns.str.strip()

    # Split dung: train = 496800 đầu, test = 449919 cuối
    # Bo qua phan o giua (449919 rows normal thừa)
    train_df = df.iloc[:496800]
    test_df  = df.iloc[-449919:]   # <-- sua o day

    train_path = os.path.join(output_dir, "normal.csv")
    test_path  = os.path.join(output_dir, "attack.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train (normal.csv): {len(train_df):,} rows")
    print(f"Test  (attack.csv): {len(test_df):,} rows")