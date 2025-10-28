import torch
import numpy as np
import json
from argparse import ArgumentParser
from data_provider import data_provider

# Choose dataset
dataset = "ETTm2"  # can change to ETTh1, ETTh2, ETTm1, etc.
pred_len = 96      # 96, 192, 336, 720 etc.

torch.manual_seed(42)
np.random.seed(42)

parser = ArgumentParser()
args = parser.parse_known_args()[0]

# Load dataset config
args_path = f'C:/Users/engs2653/Desktop/Time_series_forecasting/Resulst_file/Timeseries_fuction/datasets/{dataset}.txt'
with open(args_path, 'r') as f:
    args.__dict__ = json.load(f)

# Define paths and parameters
args.root_path = 'C:/Users/engs2653/Desktop/Time_series_forecasting/Resulst_file/Timeseries_fuction/datasets/'
args.data_path = dataset + ".csv"
args.seq_len = 96
args.pred_len = pred_len
args.batch_size = 4 if "ETT" in dataset else 32

print(f"Preparing data for {pred_len}-step predictions on the {dataset} dataset.")

data_set, data_loader = data_provider(args, flag='train')
validation_set, validation_loader = data_provider(args, flag='val')
test_set, test_loader = data_provider(args, flag='test')
