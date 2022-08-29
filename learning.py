from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, make_scorer, roc_curve, auc
import numpy as np 

def get_param_for_model(pipeline, param):
    name_model = pipeline.steps[-1][0]
    new_keys = [name_model + '__' + key for key in param.keys()]
    return dict(list(zip(new_keys, param.values())))


def nested_cross_val(X_train, y_train, model, transformer, param):
    pipeline = make_pipeline(transformer, model)

    searcher = GridSearchCV(estimator=pipeline,
                            scoring=make_scorer(f1_score, average='micro'),
                            cv=2,
                            param_grid=get_param_for_model(pipeline, param))
    searcher.fit(X_train, y_train)
    scores = cross_val_score(estimator=searcher,
                    X=X_train,
                    y=y_train,
                    scoring=make_scorer(f1_score, average='micro'),
                    cv=5)
    return scores.mean(), scores.std(), searcher.best_estimator_

def plot_roc_curve(estimator, X_test, y_test, X_train, y_train, pos_1):
    estimator.fit(X_train, y_train)
    probs = estimator.predict_proba(X_test)
    nes_col = probs[:, pos_1].reshape(-1,1)
    sum_of_other = np.delete(probs,pos_1,axis=1).sum(axis=1).reshape(-1,1)
    new_probs = np.hstack((nes_col, sum_of_other))
    new_y_test = np.where(y_test == pos_1, 1, 0)
    fpr, tpr, tresholds = roc_curve(new_y_test, new_probs[:,0], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc




