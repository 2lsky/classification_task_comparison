import matplotlib.pyplot as plt
import numpy as np
from matplotlib .colors import ListedColormap


def plot_prediction_map(X_train, y_train, X_test, y_test, classifier, resolution=0.02):
    colors = ('red', 'blue', 'cyan', 'lightgreen', 'gray')
    cmap = ListedColormap(colors[:len(np.unique(y_train))])
    feature_1_max, feature_1_min = X_train[:, 0].max() + 1, X_train[:, 0].min() - 1
    feature_2_max, feature_2_min = X_train[:, 1].max() + 1, X_train[:, 1].min() - 1
    x_1, x_2 = np.meshgrid(np.arange(feature_1_min, feature_1_max, resolution),
                           np.arange(feature_2_min, feature_2_max, resolution))
    classifier.fit(X_train, y_train)
    Z = classifier.predict(np.array([x_1.ravel(), x_2.ravel()]).T)
    Z = Z.reshape(x_1.shape)

    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.contourf(x_1, x_2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(x_1.min(), x_1.max())
    ax.set_ylim(x_2.min(), x_2.max())
    necessary_colors = colors[0:len(np.unique(y_train))]
    target_classes = np.unique(y_train)
    dict_of_colors = dict(list(zip(target_classes, necessary_colors)))
    for num, cl in enumerate(np.unique(y_train)):
        color = dict_of_colors[y_train.ravel()[num]]
        ax.scatter(X_train[y_train.ravel() == cl, 0], X_train[y_train.ravel() == cl, 1],
                   marker='x', color=color, label=str(cl) + ' Train')
    for num, cl in enumerate(np.unique(y_test)):
        color = dict_of_colors[y_test.ravel()[num]]
        ax.scatter(X_test[y_test.ravel() == cl, 0], X_test[y_test.ravel() == cl, 1],
                   marker='o', color=color, label=str(cl) + ' Test')
    ax.legend()
    plt.show()
