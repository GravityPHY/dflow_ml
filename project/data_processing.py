import numpy as np

use_data_without_messy_datum = True  # do not change >:(
use_cep = True
use_log_scale_y = False
only_use_imp_feats = False
split_test_train = False

def get_data_from_file(file_name):
    stuff = dict(np.load(file_name))
    n = 0
    for thing in stuff.keys():
        # print(str(thing)+str(stuff.get(thing).shape))
        n += 1
    X = np.zeros((stuff.get('zprs').size, n - 1))
    Y = stuff.get('zprs')
    for count, param in enumerate(stuff.keys()):
        # print(X.shape)
        if count < int(X.shape[1]):
            X.T[count] = stuff.get(param)
    Y = Y[:, 0]
    # print(X.shape)
    # print(Y.shape)
    return X, Y
def remove_annoying_variables(X:np.array,Y:np.array):
    #print('meow')
    thresh = 1.25
    bad_boy_list = [] # one might ask why bad datum are/can be gendered. In response, I say,
    # you're probably a poor physics undergrad being forced to sort through decade old spaghetti code,
    # what do you know about gender studies
    for index,element in enumerate(Y):
        if element>thresh:
            bad_boy_list.append(index)
    new_x = np.zeros((X.shape[0]-len(bad_boy_list),X.shape[1]))
    new_y = np.zeros((Y.shape[0]-len(bad_boy_list),))
    index = 0
    for count,row in enumerate(X):
        if not count in bad_boy_list:
            new_x[index] = X[count]
            new_y[index] = Y[count]
            index += 1
    #print(bad_boy_list)
    #print(new_x)
    #print(new_y)
    return new_x,new_y