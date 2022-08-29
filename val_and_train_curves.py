
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, make_scorer


def drawing_curves(train_scores, test_scores, x):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, train_mean, color='red', label='Train_score')
    ax.plot(x, test_mean, color='blue', label='Test_score')
    ax.fill_between(x,
                    train_mean + train_std,
                    train_mean - train_std,
                    alpha=0.15,
                    color='red')
    ax.fill_between(x,
                    test_mean + test_std,
                    test_mean - test_std,
                    alpha=0.15,
                    color='blue')
    ax.legend()
    plt.show()


def create_pipeline(model, transformer):
    return make_pipeline(transformer, model)


def choose_train_size(X, target, model, transformer):
    pipeline = make_pipeline(transformer, model)
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipeline,
                                                            X=X,
                                                            y=target,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            n_jobs=1,
                                                            scoring=make_scorer(f1_score,
                                                                                average='micro'))
    drawing_curves(train_scores, test_scores, train_sizes/len(X))


def plot_val_curves(X, target, model, trans, param_range, param_name):
    pipeline = make_pipeline(trans, model)
    train_scores, test_scores = validation_curve(estimator=pipeline,
                                                 X=X,
                                                 y=target,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=10,
                                                 scoring=make_scorer(f1_score,
                                                                     average='micro'))
    drawing_curves(train_scores, test_scores, param_range)


#choose_train_size(X_train, y_train, model, transformer)
# for randomforest
#[10-20] max_depth
#[3,5,10] n_estimators
#[2,5,10,15] min_samples_split
#[2,3,5] min_samples_leaf

#for gradient boosting
#[0.1 to 0.15] lambda
#[30-50] min_samles_split
#[15-18] min_samples_leaf
#[75-100] n_estimators
#[10-15] max_depth 