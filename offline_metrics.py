import os
import constants as c
from functions import destandardize_field
import mlflow
import torch
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
from functions import *
from DoWnGAN.losses import *

# test
import matplotlib.pyplot as plt

batch_size = 64

# Define experiment regions
regions = [c.florida, c.central, c.west]
region_names = ["florida", "central", "west"]
# Define experiments shortnames
exps = ["CNN", "13x13", "9x9", "5x5", "nf"]

# Define partitions
sets = ["train", "validation"]
with torch.no_grad():
    for s in sets:
        for rname, rdict  in zip(region_names, regions):
            print(rname)
            # Define train data including complete covariates set
            coarse = xr.open_dataset(f"{os.getenv('GAN_DATA')}/ground_truth/coarse_{s}_{rname}.nc").to_array()[0, ...]

            # Randomize
            randoms = np.random.randint(0, coarse.time.shape[0], coarse.time.shape[0])
            coarse_torch = torch.from_numpy(coarse.values[randoms, ...]).float()

            # Load the fine scale GT
            dsgt = xr.open_dataset(f"{os.getenv('GAN_DATA')}/ground_truth/wrf_{s}_{rname}.nc").to_array()[0, ...]
            fine_torch = torch.from_numpy(dsgt.values[randoms, ...]).float()
            aranged = torch.split(torch.arange(0, coarse_torch.size(0)), int(coarse_torch.size(0)/batch_size), dim=0)

            print("Chunk size: ", aranged[0].size(0))

            complete = {}
            for experiment in rdict.keys():     # Iterate through experiments
                long_mae = []
                long_mse = []
                long_wass =[]
                long_msssim = []
                hash_dict = {}
                enum, code = rdict[experiment] # Extract the details of the experiment from constants.py
                print(code)
                for i in tqdm(np.arange(0, 1000, 100)): # Iterate through each of the saved models
                    epoch_metrics = {
                        "MAE": [],
                        "MSE": [],
                        "MSSSIM": [],
                        "WASS": []
                    }
                    # Paths to models
                    g_path = f"{os.getenv('EXPERIMENT_DATA_DIR')}/{enum}/{code}/artifacts/Generator/Generator_{i}"
                    c_path = f"{os.getenv('EXPERIMENT_DATA_DIR')}/{enum}/{code}/artifacts/Critic/Critic_{i}"
                    
                    # Load models
                    G = mlflow.pytorch.load_model(g_path)
                    state_dict = mlflow.pytorch.load_state_dict(g_path)
                    G.load_state_dict(state_dict)

                    C = mlflow.pytorch.load_model(c_path)
                    state_dict = mlflow.pytorch.load_state_dict(c_path)
                    C.load_state_dict(state_dict)

                    # Work way through chunked inputs@
                    for r in aranged:
                        Y = G(coarse_torch[r, ...].to(c.device))
                        X = fine_torch[r, ...].to(c.device)
                        
                        fake_list = [Y[:, 0, ...], Y[:, 1, ...]]
                        real_list = [X[:, 0, ...], X[:, 1, ...]]

                        # Calculate Wasserstein metrics
                        C_real = torch.mean(C(X - c.filters[experiment](c.padding[experiment](X)).to(c.device)))
                        C_fake = torch.mean(C(Y - c.filters[experiment](c.padding[experiment](Y)).to(c.device)))
                        wass = C_real - C_fake
                        epoch_metrics["WASS"].append(wass.detach().cpu())

                        # De-standardize data
                        mu, su = c.region_stats_u10[rname]
                        mv, sv = c.region_stats_v10[rname]
                        Y[:, 0, ...] = Y[:, 0, ...]*su + mu
                        Y[:, 1, ...] = Y[:, 1, ...]*sv + mv

                        X[:, 0, ...] = X[:, 0, ...]*su + mu
                        X[:, 1, ...] = X[:, 1, ...]*sv + mv

                        mae = content_loss(X, Y, c.device).item()
                        epoch_metrics["MAE"].append(mae)

                        mse = content_MSELoss(X, Y, c.device).item()
                        epoch_metrics["MSE"].append(mse)

                        msssim = SSIM_Loss(X, Y, c.device).item()
                        epoch_metrics["MSSSIM"].append(msssim)

                    long_mae.append((np.min(epoch_metrics["MAE"]), np.mean(epoch_metrics["MAE"]), np.max(epoch_metrics["MAE"])))
                    long_mse.append((np.min(epoch_metrics["MSE"]), np.mean(epoch_metrics["MAE"]), np.max(epoch_metrics["MAE"])))
                    long_msssim.append((np.min(epoch_metrics["MSSSIM"]), np.mean(epoch_metrics["MAE"]), np.max(epoch_metrics["MAE"])))
                    long_wass.append((np.min(epoch_metrics["WASS"]), np.mean(epoch_metrics["MAE"]), np.max(epoch_metrics["MAE"])))

                hash_dict["mae"] = long_mae
                hash_dict["mse"] = long_mse
                hash_dict["msssim"] = long_msssim
                hash_dict["wasserstein"] = long_wass 
                complete[code] = hash_dict

            # Write to file
            df = pd.DataFrame()
            metric = ["mae", "mse", "msssim", "wasserstein"]
            for m in metric:
                for key in complete.keys():
                    df[f"{key}_{m}_min"] = [t[0] for t in complete[key][m]]
                    df[f"{key}_{m}_mean"] = [t[1] for t in complete[key][m]]
                    df[f"{key}_{m}_max"] = [t[2] for t in complete[key][m]]

            df.to_csv(f"offline_metrics/{rname}_{s}_offline_metrics.csv")
            print("Logged to csv file!")
            torch.cuda.empty_cache()
