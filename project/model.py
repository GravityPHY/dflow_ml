import numpy as np
from scipy import linalg
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_processing import *


class weighted_krr:
    def __init__(self,weight):
        """

        :param weight:
        """
        self.w=np.diagflat(weight)
    def get_Kernel(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        if y is None:
            Kernel=np.matmul(X@self.w,X.T)
            np.fill_diagonal(Kernel,80)
        else:
            Kernel=np.matmul(X@self.w,y.T)

        return Kernel

    def fit(self, X,y):
        """

        :param X:
        :param y:
        :return:
        """
        kernel=self.get_Kernel(X,None)
        self.alpha = np.linalg.inv(kernel) @ y
        #return self.alpha

    def predict(self,  train_X,test_X):
        """

        :param train_X:
        :param test_X:
        :return:
        """
        K=self.get_Kernel( train_X,test_X)
        return np.dot(K.T, self.alpha)


def linear_regression():
    fig, ax = plt.subplots()
    ax.legend()
    ax.set_xlabel('ZPRS Predicted')
    ax.set_ylabel('ZPRS Original')
    ax.set_title('Linear Regression Results w/out ZPRS over 1.25')
    X,y = get_data_from_file("./data/htzg_dataset_oncv.npz")

    if use_data_without_messy_datum:
        X,y = remove_annoying_variables(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.1)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_guess = np.zeros((y_test.shape[0],))
    for count,data in enumerate(X_test):
        y_guess[count] = regr.predict([data])

def my_kernel(random=10,size=0.2):

    X,y = get_data_from_file("./data/htzg_dataset_oncv.npz")
    if use_data_without_messy_datum:
        X,y = remove_annoying_variables(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random, test_size=size)
    my_krr.fit(X_train, y_train)
    y_guess = my_krr.predict(X_train, X_test)

    print(mean_squared_error(y_test, y_guess))
    y_train_guess = my_krr.predict(X_train, X_train)
    print(mean_squared_error(y_train, y_train_guess))
    slope, intercept, r, p, std_err = stats.linregress(y_train_guess, y_train)

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, y_train_guess))

    fig, ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(4)
    ax.legend(['Blue - Training Data', 'Yellow - Test Data'])
    ax.set_xlabel('ZPRS Predicted')
    ax.set_ylabel('ZPRS Original')
    ax.set_title('New kernel Regression')
    ax.scatter(y_train_guess, y_train, c=['#1f77b4'])
    ax.scatter(y_guess, y_test, c=['#bcbd22'])
    ax.plot(y_test, y_test, color='yellow')

    # print("Slope: " + str(slope))

    plt.savefig('./pred-test-mykernel.jpg')


