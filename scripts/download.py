import datasets

ds = datasets.load_dataset("autogluon/chronos_datasets_extra", "ETTh", split="train", trust_remote_code=True)
# ds.set_format("numpy")  # sequences returned as numpy arrays
ds = datasets.load_dataset("autogluon/chronos_datasets_extra", "ETTm", split="train", trust_remote_code=True)
