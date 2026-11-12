from msi.msi import MSI
from diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import networkx as nx

msi = MSI()
msi.load()

dp = DiffusionProfiles(alpha = 0.9, max_iter = 1000, tol = 1e-06, weights = {'down_biological_function': 4, 'indication': 3.5, 
'biological_function': 6, 'up_biological_function': 2, 'protein': 4, 'drug': 3.5}, num_cores = 4, 
save_load_file_path = "results/")

dp.calculate_diffusion_profiles(msi) 
