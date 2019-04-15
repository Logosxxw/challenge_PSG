import pandas as pd
import numpy as np
import pickle
import os
from lxml import etree
from logos_tools import *
from logos_opta import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, GRU
from tensorflow.keras.models import Model

model_dir = 'model'

# 获取所有团队样本
use_dir = '../resources/samples/team/'
team_samples = [f for f in os.listdir(use_dir) if 
os.path.isfile(os.path.join(use_dir, f)) and '.DS' not in f]

# 读取样本
data_list = []
for x in team_samples:
#     print(x)
    data = pd.read_csv(os.path.join(use_dir, x), sep='\t')
    data['game'] = x.split('.')[0]
    data_list.append(data)
data = pd.concat(data_list, axis=0, sort=False)
data['last_position'] = data.last_event.map(lambda x:x.split('-')[2])

# 字段整理
self_fields = {'pass', 'pass1_rate', 'front', 'positionC_rate', 'positionL_rate', 'positionR_rate', 'positionB_rate', 
               'short_rate', 'middle_rate', 'long_rate', 'takeon', 'ballTouch', 'clearance', 'foul'}
opp_fields = self_fields.copy()
useful_fields = {'period', 'last_team', 'last_position', 'last_event_type', 'last_x', 'last_y', 'last10_list', 'game'}
label_fields = {'y_team', 'y_x', 'y_y'}

# 合并主客队对偶的特征
data.index = data.game+','+data.start.astype('str')+','+data.end.astype('str')
home_df = data.loc[data.team_id==1, self_fields.union(useful_fields).union(label_fields)]
away_df = data.loc[data.team_id==0, opp_fields]
away_df.columns = away_df.columns + '_o'
data_new = pd.concat([home_df, away_df], axis=1, sort=False)
opp_fields = set(away_df.columns)

# 获取去重的比赛列表
data_new = data_new.reset_index(drop=True)
games_s = data_new.game
games = games_s.unique()
rounds = [int(x.split(',')[0].split('_')[0]) for x in games]
homes = [x.split(',')[0].split('_')[1] for x in games]
aways = [x.split(',')[0].split('_')[3] for x in games]
games_df = pd.DataFrame({'rounds':list(rounds), 'homes':list(homes), 'aways':list(aways)})

# 特征变换
team_id_features = self_fields.union(opp_fields).union(useful_fields)-{'last10_list', 'game', 'last_x', 'last_y'}
X_id = data_new.loc[:, team_id_features]
y_id = data_new.loc[:, 'y_team']

cat_features = {'period', 'last_position', 'last_event_type'}
column_trans = ColumnTransformer(
    [('onehot', OneHotEncoder(handle_unknown='ignore'), list(cat_features)),
     ('standard', StandardScaler(), list(team_id_features-cat_features))], 
    remainder='drop')
X_id = column_trans.fit_transform(X_id)
transformer_file = 'column_transformer_rnn'
with open(os.path.join(model_dir, transformer_file), 'wb') as handle:
    pickle.dump(column_trans, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 序列处理
docs_seg = [x.split(',') for x in data_new.last10_list]
t = keras.preprocessing.text.Tokenizer(oov_token=1)
t.fit_on_texts(docs_seg)
tokenizer_file = 'tokenizer'
with open(os.path.join(model_dir, tokenizer_file), 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 转化为word index
encoded_docs = t.texts_to_sequences(docs_seg)
vocabulary_size = len(t.word_index)+1
# 填充为固定长度
sequence_length = 10
padded_docs = keras.preprocessing.sequence.pad_sequences(
    encoded_docs, maxlen=sequence_length, padding='pre')

