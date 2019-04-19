import pandas as pd
import numpy as np
import pickle
import os
from lxml import etree
from logos_tools import *
from logos_opta import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost.sklearn import XGBClassifier, XGBRegressor
from absl import flags, logging, app

FLAGS = flags.FLAGS
flags.DEFINE_string('sample_dir', None, 'sample_dir')
flags.DEFINE_string('model_dir', None, 'model_dir')
flags.DEFINE_string('dim_game', None, 'dim_game')
flags.DEFINE_string('transformer_file', None, 'transformer_file')
flags.DEFINE_string('model_file', None, 'model_file')
flags.DEFINE_integer('trees', None, 'xgb max trees')
flags.DEFINE_integer('patience', None, 'early stopping patience')
flags.DEFINE_integer('xgb_n_jobs', None, 'xgb_n_jobs')
flags.DEFINE_string('depth_search', None, 'depth_search')
flags.DEFINE_string('gamma_search', None, 'gamma_search')
flags.DEFINE_string('alpha_search', None, 'alpha_search')

def main(_):
    sample_dir = FLAGS.sample_dir
    model_dir = FLAGS.model_dir
    dim_game = FLAGS.dim_game
    transformer_file = FLAGS.transformer_file
    model_file = FLAGS.model_file
    trees = FLAGS.trees
    patience = FLAGS.patience
    xgb_n_jobs= FLAGS.xgb_n_jobs
    depth_search = FLAGS.depth_search
    gamma_search = FLAGS.gamma_search
    alpha_search = FLAGS.alpha_search
    estimator = XGBClassifier
    random_state = 7
    objective = 'multi:softprob'
    scoring='accuracy'
    eval_metric='merror'
    depth_list = string_param_to_num_list(depth_search, int)
    gamma_list = string_param_to_num_list(gamma_search, float)
    alpha_list = string_param_to_num_list(alpha_search, float)
    test_rounds = [19]
    sub_sample_step = 10

    # 获取所有player样本
    samples = [f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f)) and '.DS' not in f]

    # 读取样本
    data_list = []
    for x in samples:
        data = pd.read_csv(os.path.join(sample_dir, x), sep='\t')
        data['game'] = x.split('.')[0]
        data_list.append(data)
    data = pd.concat(data_list, axis=0, sort=False)

    # 削减样本防止过拟合
    data.index = data.game
    data = data.groupby(data.index).apply(sub_sample, step=sub_sample_step)
    data = data.reset_index(drop=True)

    # 特征变换
    all_fields = set(data.columns)
    not_num_fields = {'period', 'team_id', 'last20_list', 'player_id', 'player_name', 'position_use', 'start', 'end',
                'team_id_real', 'team_name', 'game'}
    use_str_fields = {'period', 'team_id'}
    column_trans = ColumnTransformer(
        [('onehot', OneHotEncoder(handle_unknown='ignore'), list(use_str_fields)),
        ('standard', StandardScaler(), list(all_fields-not_num_fields))], 
        remainder='drop')
    X = column_trans.fit_transform(data)
    # 保存transformer
    with open(os.path.join(model_dir, transformer_file), 'wb') as handle:
        pickle.dump(column_trans, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 选取某几轮作为测试集其余作为训练集
    train_index, test_index = get_train_test_index_from_rounds(dim_game, data.game, test_rounds)
    y = data.player_id
    X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]
    # grid search cv
    eval_set = [(X_train, y_train), (X_valid, y_valid)]
    data['test_fold'] = -1
    data.loc[test_index, 'test_fold'] = 0
    test_fold = data.test_fold

    # 训练模型
    model = train_XGB(estimator, X, y, test_fold, eval_set, eval_metric, scoring, xgb_n_jobs, random_state, trees, patience, objective, depth_list, gamma_list, alpha_list)

    # 保存model
    with open(os.path.join(model_dir, model_file), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    app.run(main)