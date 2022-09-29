import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from model import *
from plot import *
# load .npz data
data_address = "./project/data/"
#data_address = "htzg_dataset_oncv.npz" # if run at local machine
data = np.load(data_address)

# data processing
#train_properties = data.files[0:-1]
#X = np.hstack([data[i].reshape(-1, 1) for i in train_properties])  # same dimensionality before stacking
#y = data[data.files[-1]]
#train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=848)


my_kernel()

# model
# model training

#my_kernel = compute_kernel(train_X, train_y, kernel_function)
# predict
#predict_y = kernel_model_pred(train_y, my_kernel, kernel_function, train_X, test_X)
# save figure
#plot_save_figure(predict_y, test_y)
# save reuslt
#np.savez("result", predict_y)

