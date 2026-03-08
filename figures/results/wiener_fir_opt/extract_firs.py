import sys, os

sys.path.append(r"../../lib")

import plot_lib as pl

import matplotlib.pyplot as plt
import numpy as np
import torch

curr_path = os.getcwd()

fir_folder_path = os.path.join(curr_path, "data", "wiener_branch_4_taps_11_no_power_ramp_lr_10")

weights_path = os.path.join(fir_folder_path, "weights_best.pt")

scale_factor = 16384 ** 4

# x_test = np.load(os.path.join(fir_folder_path, "x_test.npy"))
# d_test = np.load(os.path.join(fir_folder_path, "d_test.npy")) / scale_factor
# y_test = np.load(os.path.join(fir_folder_path, "y_test.npy")) / scale_factor

weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)

branch_num = 4
filter_num = 3

filters = {}
for j_branch in range(branch_num):
    filters.update({f"branch_{j_branch}": {}})
    for j_filter in range(filter_num):
        curr_filter_real = weights[f"fir_cells.{j_branch}.{j_filter}.weight"].data.detach().cpu().numpy().real
        curr_filter_imag = weights[f"fir_cells.{j_branch}.{j_filter}.weight"].data.detach().cpu().numpy().imag
        filters[f"branch_{j_branch}"].update({f"filter_{j_filter}": {}})
        filters[f"branch_{j_branch}"][f"filter_{j_filter}"].update({"real": curr_filter_real})
        filters[f"branch_{j_branch}"][f"filter_{j_filter}"].update({"imag": curr_filter_imag})
        np.save(os.path.join(curr_path, "data", "opt_filters", "initial", f"fir_branch_{j_branch}_num_{j_filter}_real.npy"), curr_filter_real)
        np.save(os.path.join(curr_path, "data", "opt_filters", "initial", f"fir_branch_{j_branch}_num_{j_filter}_imag.npy"), curr_filter_imag)
