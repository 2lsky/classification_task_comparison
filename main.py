
from analysis_dataset import data, analysis_dataframe
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   LabelEncoder)
from sklearn.compose import ColumnTransformer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier

from learning import nested_cross_val, plot_roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import numpy as np 


dataframe = analysis_dataframe(data)
target = dataframe.target
encoder_for_target = LabelEncoder()
target = encoder_for_target.fit_transform(target)
X = dataframe.X
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    target,
                                                    test_size=0.1)
num_cols = dataframe.num_features()
cat_cols = dataframe.cat_features()
bin_cols = dataframe.binary_features()
transformer = ColumnTransformer(transformers=[('standard',
                                               StandardScaler(),
                                               num_cols),
                                              ('encoder',
                                               OneHotEncoder(),
                                               cat_cols),
                                              ('dim_reducer',
                                               LinearDiscriminantAnalysis(n_components=1),
                                               np.hstack((num_cols, bin_cols)))],
                                remainder='passthrough')
models = [RandomForestClassifier(), GradientBoostingClassifier()]
params = [{'max_depth': range(10, 20), 'n_estimators': range(3, 10)},
          {'n_estimators': range(10, 20), 'max_depth': range(10, 15)}]
colors = ['red', 'green']


fig, ax = plt.subplots(ncols=1, nrows=1)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')

for index, model in enumerate(models):
    mean_score, std_score, best_estimator = nested_cross_val(X_train,
                                                             y_train,
                                                             model,
                                                             transformer,
                                                             params[index])
    print(f'F_score - {mean_score} +/- {std_score} for Model_{index}')
    mean_tpr = 0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(0, len(np.unique(target))):
        fpr, tpr, roc_auc = plot_roc_curve(best_estimator,
                                           X_test,
                                           y_test,
                                           X_train,
                                           y_train,
                                           pos_1=i)
        ax.plot(fpr, tpr, color='gray', marker='.')
        mean_tpr += np.interp(mean_fpr, fpr, tpr) 
        mean_tpr[0] = 0.0
    mean_tpr /= len(np.unique(target))
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr,
            mean_tpr,
            color=colors[index],
            label=f'Model_{index}_mean_AUC - {mean_auc}')
ax.legend()
plt.show()
