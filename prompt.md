!python run_swat.py
Kaggle input path not found. using local path if available.
Warning: normal.csv or attack.csv not found in /kaggle/working/carla_resnet/datasets/swat
Please ensure 'datasets/swat/normal.csv' and 'datasets/swat/attack.csv' exist.
Set swat_DATASET_PATH to /kaggle/working/carla_resnet/datasets/swat

==============================
STARTING EXPERIMENTS
==============================
GPU available: Tesla T4

Running dataset: swat
Error running pretext for swat: Command '['/usr/bin/python3', 'carla_pretext.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/pretext/carla_pretext_swat.yml', '--fname', 'swat']' returned non-zero exit status 1.
Traceback (most recent call last):
  File "/kaggle/working/carla_resnet/carla_pretext.py", line 240, in <module>
    main()
  File "/kaggle/working/carla_resnet/carla_pretext.py", line 143, in main
    train_dataset = get_train_dataset(p, train_transforms, sanomaly, to_augmented_dataset=True)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/carla_resnet/utils/common_config.py", line 118, in get_train_dataset
    dataset = SWAT(p['fname'], train=True, transform=transform, sanomaly=sanomaly,
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/carla_resnet/data/SWAT.py", line 45, in __init__
    temp = pd.read_csv(file_path)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pandas/io/common.py", line 873, in get_handle
    handle = open(
             ^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/carla_resnet/datasets/swat/normal.csv'

Error running classification for swat: Command '['/usr/bin/python3', 'carla_classification.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/classification/carla_classification_swat.yml', '--fname', 'swat']' returned non-zero exit status 1.
Traceback (most recent call last):
  File "/kaggle/working/carla_resnet/carla_classification.py", line 213, in <module>
    main()
  File "/kaggle/working/carla_resnet/carla_classification.py", line 51, in main
    train_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/carla_resnet/utils/common_config.py", line 172, in get_aug_train_dataset
    data_dict = torch.load(p['contrastive_dataset'], weights_only=False)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1500, in load
    with _open_file_like(f, "rb") as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 768, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 749, in __init__
    super().__init__(open(name, mode))  # noqa: SIM115
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'results/swat/swat/pretext/con_train_dataset.pth'

Max GPU Memory after swat: 0.00 MB

==============================
DONE ALL SWAT DATASETS
Total time: 18.88 s
Avg / dataset: 18.88 s
==============================

Time results saved to results/swat/time_results.json

==============================
STARTING EVALUATION (PAPER STYLE)
==============================
Skip swat (missing files)
No results!