import matplotlib.pyplot as plt


def plot_save_figure(y_hat, test_y):
    plt.plot(y_hat, test_y, 'ro')
    plt.savefig('./pred-test.jpg')


