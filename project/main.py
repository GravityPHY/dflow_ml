import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from model import *

# load .npz data
data_address = "./project/data/"
#data_address = "htzg_dataset_oncv.npz" # if run at local machine
data = np.load(data_address)



my_kernel()



