import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, PredefinedSplit

def safe_convert(x, f, default):
    try: 
        return f(x)
    except:
        return default

def handle_divided_by0(x,y):
    return round(x/y,6) if y>0 else 0

string_param_to_num_list = lambda param, f: [f(x.strip()) for x in param.split(',')]

def train_XGB(estimator, X, y, test_fold, eval_set, eval_metric, scoring, xgb_n_jobs, random_state, trees, patience, 
objective, depth_list, gamma_list, alpha_list, refit_on_all=False):
    # grid search
    result = GridSearchCV(estimator=estimator(n_estimators=trees, objective=objective, random_state=random_state, n_jobs=xgb_n_jobs), 
                        param_grid={'max_depth':depth_list, 
                                    'gamma':gamma_list, 
                                    'reg_alpha':alpha_list}, 
                        scoring=scoring, 
                        cv=PredefinedSplit(test_fold=test_fold), 
                        refit=False, 
                        n_jobs=1,
                        iid=False, 
                        return_train_score=True, 
                        error_score=np.nan, 
                        verbose=1)
    xgb = result.fit(X, y, eval_set=eval_set, eval_metric=eval_metric, early_stopping_rounds=patience, verbose=10)
    # 打印gridsearch结果
    result_dict = {k:np.round(result.cv_results_[k],4) for k in ['mean_fit_time', 'mean_train_score', 'mean_test_score']}
    result_dict.update({'params':result.cv_results_['params']})
    print(pd.DataFrame(result_dict))
    # 用最佳参数重新在整个数据集上训练模型
    model = estimator(n_estimators=trees, 
                        objective=objective, 
                        n_jobs=xgb_n_jobs, 
                        random_state=random_state, 
                        max_depth=xgb.best_params_['max_depth'], 
                        gamma=xgb.best_params_['gamma'], 
                        reg_alpha=xgb.best_params_['reg_alpha'])
    if (refit_on_all):
        model.fit(X, y, eval_set=eval_set, eval_metric=eval_metric, early_stopping_rounds=patience)
    else:
        model.fit(eval_set[0][0], eval_set[0][1], eval_set=eval_set, eval_metric=eval_metric, early_stopping_rounds=patience, verbose=False)
    return model
