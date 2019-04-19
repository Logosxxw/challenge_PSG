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
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression

EVENT_FILE = 'event.txt'
QUALIFIER_FILE = 'qualifier.txt'
ASSOCIATE_FILE = 'associate_use.tsv'
GAME_FILE = 'test.xml'
PLAYER_FILE = 'players.tsv'
player_transformer_file = 'player_transformer.pkl'
player_model_file = 'player_model.pkl'
next_team_transformer_file = 'next_team_transformer.pkl'
next_team_model_file = 'next_team_model.pkl'
next_x_transformer_file = 'next_x_transformer.pkl'
next_x_model_file = 'next_x_model.pkl'
next_y_transformer_file = 'next_y_transformer.pkl'
next_y_model_file = 'next_y_model.pkl'
USE_PLAYERID = '1'

def get_player_sample(use_df, start, end):
    use_half = '2' if start > 45*60 else '1'
    # delete last 10 event and stat
    use_df = use_df.iloc[:-10]
    # game period
    use_period = get_period(use_half, start)
    # shoot stat
    shoot_df = pd.DataFrame({'bigChance':[0], 'bigChance_rate':[0.0], 'head':[0], 'head_rate':[0.0], 
                             'inside':[0], 'inside_rate':[0.0], 'onTarget':[0], 'onTarget_rate':[0.0], 
                             'outside':[0], 'outside_rate':[0.0], 'shoot':[0]})
    selected = use_df.loc[use_df.event_type.isin(['Miss', 'Post', 'Attempt Saved', 'Goal'])]
    selected = selected.loc[selected.player_id==USE_PLAYERID]
    if len(selected)>0:
        shoot_df = shoot_stat(selected)
    # pass stat
    pass_df = pd.DataFrame({'pass':[0], 'pass1_rate':[0.0], 
                     'front':[0], 'front_rate':[0.0], 'front1_rate':[0.0],
                     'key':[0], 'key_rate':[0.0],
                     'cross':[0], 'cross_rate':[0.0], 'cross1_rate':[0.0],
                     'through':[0], 'through_rate':[0.0], 'through1_rate':[0.0],
                     'chipped':[0], 'chipped_rate':[0.0], 'chipped1_rate':[0.0],
                     'forward':[0], 'forward_rate':[0.0], 'forward1_rate':[0.0],
                     'back':[0], 'back_rate':[0.0], 'back1_rate':[0.0],
                     'left':[0], 'left_rate':[0.0], 'left1_rate':[0.0],
                     'right':[0], 'right_rate':[0.0], 'right1_rate':[0.0],
                     'short':[0], 'short_rate':[0.0], 'short1_rate':[0.0],
                     'middle':[0], 'middle_rate':[0.0], 'middle1_rate':[0.0],
                     'long':[0], 'long_rate':[0.0], 'long1_rate':[0.0],
                     'positionC':[0], 'positionC_rate':[0.0], 'positionC1_rate':[0.0],
                     'positionL':[0], 'positionL_rate':[0.0], 'positionL1_rate':[0.0],
                     'positionR':[0], 'positionR_rate':[0.0], 'positionR1_rate':[0.0],
                     'positionB':[0], 'positionB_rate':[0.0], 'positionB1_rate':[0.0]
                    })
    selected = use_df.loc[use_df.event_type=='Pass']
    selected = selected.loc[selected.player_id==USE_PLAYERID]
    if (len(selected)>0):
        pass_df = pass_stat(selected)
    # other stat
    selected = use_df.loc[use_df.player_id==USE_PLAYERID]
    other_df = other_stat(selected)
    # get team_id
    team_id_s = pd.Series(selected.team_id.iloc[0], name='team_id')
    # concat
    sample = pd.concat([pass_df, shoot_df, other_df, team_id_s], axis=1, sort=False)
    sample = sample.fillna(0)
    sample['period'] = use_period
    sample['start'] = start
    sample['end'] = end
    return sample

def get_team_sample(use_df, start, end):
    # game period
    use_half = '2' if start > 45*60 else '1'
    use_period = get_period(use_half, start)
    # last 10 event
    last10 = use_df.iloc[-10:]
    last10_list = list(last10.team_id +'-'+ last10.event_type +'-'+ last10.apply(lambda row: get_position_xy(row.x, row.y), axis=1) +'-'+ use_period)
    last_team = last10.team_id.iloc[-1]
    last_event_type = last10.event_type.iloc[-1]
    last_x = safe_convert(last10.x.iloc[-1], float, 0)
    last_y = safe_convert(last10.y.iloc[-1], float, 0)
    last_event = last10_list[-1]
    # delete last 10 event and stat
    use_df = use_df.iloc[:-10]
    # shoot stat
    shoot_df = pd.DataFrame({'bigChance':[0,0], 'bigChance_rate':[0.0,0.0], 'head':[0,0], 'head_rate':[0.0,0.0], 'inside':[0,0], 'inside_rate':[0.0,0.0], 'onTarget':[0,0], 'onTarget_rate':[0.0,0.0], 'outside':[0,0], 'outside_rate':[0.0,0.0], 'shoot':[0,0]}, index=['0','1'])
    shoot_df.index.name = 'team_id'
    selected = use_df.loc[use_df.event_type.isin(['Miss', 'Post', 'Attempt Saved', 'Goal'])]
    if len(selected)>0:
        shoot_df = selected.groupby('team_id').apply(shoot_stat)
        shoot_df = shoot_df.reset_index(level=1, drop=True)
    # pass stat
    pass_df = pd.DataFrame({'pass':[0,0], 'pass1_rate':[0.0,0.0], 
                         'front':[0,0], 'front_rate':[0.0,0.0], 'front1_rate':[0.0,0.0],
                         'key':[0,0], 'key_rate':[0.0,0.0],
                         'cross':[0,0], 'cross_rate':[0.0,0.0], 'cross1_rate':[0.0,0.0],
                         'through':[0,0], 'through_rate':[0.0,0.0], 'through1_rate':[0.0,0.0],
                         'chipped':[0,0], 'chipped_rate':[0.0,0.0], 'chipped1_rate':[0.0,0.0],
                         'forward':[0,0], 'forward_rate':[0.0,0.0], 'forward1_rate':[0.0,0.0],
                         'back':[0,0], 'back_rate':[0.0,0.0], 'back1_rate':[0.0,0.0],
                         'left':[0,0], 'left_rate':[0.0,0.0], 'left1_rate':[0.0,0.0],
                         'right':[0,0], 'right_rate':[0.0,0.0], 'right1_rate':[0.0,0.0],
                         'short':[0,0], 'short_rate':[0.0,0.0], 'short1_rate':[0.0,0.0],
                         'middle':[0,0], 'middle_rate':[0.0,0.0], 'middle1_rate':[0.0,0.0],
                         'long':[0,0], 'long_rate':[0.0,0.0], 'long1_rate':[0.0,0.0],
                         'positionC':[0,0], 'positionC_rate':[0.0,0.0], 'positionC1_rate':[0.0,0.0],
                         'positionL':[0,0], 'positionL_rate':[0.0,0.0], 'positionL1_rate':[0.0,0.0],
                         'positionR':[0,0], 'positionR_rate':[0.0,0.0], 'positionR1_rate':[0.0,0.0],
                         'positionB':[0,0], 'positionB_rate':[0.0,0.0], 'positionB1_rate':[0.0,0.0]
                        }, index=['0','1'])
    pass_df.index.name = 'team_id'
    selected = use_df.loc[use_df.event_type=='Pass']
    if (len(selected)>0):
        pass_df = selected.groupby('team_id').apply(pass_stat)
        pass_df = pass_df.reset_index(level=1, drop=True)
    # other stat
    other_df = use_df.groupby('team_id').apply(other_stat)
    other_df = other_df.reset_index(level=1, drop=True)
    # concat
    sample = pd.concat([pass_df, shoot_df, other_df], axis=1, sort=False)
    sample = sample.fillna(0)
    sample['period'] = use_period
    sample['last_team'] = last_team
    sample['last_event_type'] = last_event_type
    sample['last_x'] = last_x
    sample['last_y'] = last_y
    sample['last_event'] = last_event
    sample['last10_list'] = ','.join(last10_list)
    sample['team_id'] = sample.index
    sample['start'] = start
    sample['end'] = end
    # add new features
    data = add_new_features(sample)
    return data

def concat_home_and_away_team(data):
    # concat home and away
    self_fields = {'pass1', 'front1', 'defend', 'ballTouch', 'shoot', 'foul', 'clearance', 'takeon1',
                  'positionC1', 'positionL1', 'positionR1', 'positionB1'}
    opp_fields = self_fields.copy()
    useful_fields = {'period', 'last_team', 'last_position', 'last1_event', 'last2_event', 'last_x', 'last_y', 
                     'last10_list', 'game'}
    home_df = data.loc[data.team_id=='1', list(self_fields.union(useful_fields))]
    home_df.reset_index(drop=True, inplace=True)
    away_df = data.loc[data.team_id=='0', list(opp_fields)]
    away_df.reset_index(drop=True, inplace=True)
    away_df.columns = away_df.columns + '_o'
    data_new = pd.concat([home_df, away_df], axis=1, sort=False)
    return data_new.reset_index(drop=True)  

def concat_home_and_away_xy(data):
    # concat home and away
    self_fields = {'pass1', 'front1', 'defend', 'ballTouch', 'shoot', 'foul', 'clearance', 'takeon1',
              'positionC1', 'positionL1', 'positionR1', 'positionB1',
              'positionC_rate', 'positionL_rate', 'positionR_rate', 'positionB_rate', 
               'short_rate', 'middle_rate', 'long_rate'}
    opp_fields = self_fields.copy()
    useful_fields = {'period', 'last_team', 'last_position', 'last1_event', 'last2_event', 'last10_list', 'game',
                    'last_x', 'last_y'}
    add_pisition_fields = {'position9', 'position8', 'position7', 'position6', 'position5'}
    useful_fields = useful_fields.union(add_pisition_fields)
    home_df = data.loc[data.team_id=='1', list(self_fields.union(useful_fields))]
    home_df.reset_index(drop=True, inplace=True)
    away_df = data.loc[data.team_id=='0', list(opp_fields)]
    away_df.reset_index(drop=True, inplace=True)
    away_df.columns = away_df.columns + '_o'
    data_new = pd.concat([home_df, away_df], axis=1, sort=False)
    return data_new.reset_index(drop=True)    

# load dim file
event_df = pd.read_csv(EVENT_FILE, sep='|')
event_s = pd.Series(data=event_df.type.values, index=[str(x) for x in event_df.id])
event_dict = event_s.to_dict()
qualifier_df = pd.read_csv(QUALIFIER_FILE, sep='|')
qualifier_s = pd.Series(data=qualifier_df.type.values, index=[str(x) for x in qualifier_df.id])
qualifier_dict = qualifier_s.to_dict()
associate_df = pd.read_csv(ASSOCIATE_FILE, sep='\t')
associate_dict = {str(associate_df.Type_id[i]):associate_df.qualifier_id[i].split(',') for i in range(len(associate_df))}

# load models
with open(player_transformer_file, 'rb') as handle:
    player_column_trans = pickle.load(handle)
with open(player_model_file, 'rb') as handle:
    player_model = pickle.load(handle)
with open(next_team_transformer_file, 'rb') as handle:
    next_team_column_trans = pickle.load(handle)
with open(next_team_model_file, 'rb') as handle:
    next_team_model = pickle.load(handle)
with open(next_x_transformer_file, 'rb') as handle:
    next_x_column_trans = pickle.load(handle)
with open(next_x_model_file, 'rb') as handle:
    next_x_model = pickle.load(handle)
with open(next_y_transformer_file, 'rb') as handle:
    next_y_column_trans = pickle.load(handle)
with open(next_y_model_file, 'rb') as handle:
    next_y_model = pickle.load(handle)

def Result(xml_1):
    game = xml_1.xpath('Game')[0]
    # xml to dataframe
    game_df = pd.concat([parse_event(x, event_dict, qualifier_dict, associate_dict) for x in game], axis=0)
    game_df = game_df.fillna(value=UNK)
    # cleaning
    game_df.loc[game_df.event_type=='Deleted event', 
                ['length', 'direction', 'position', 'qualifier']] = UNK
    game_df.loc[game_df.event_type=='Clearance', 
                ['direction', 'position']] = UNK
    game_df['time'] = game_df['min'].astype('int')*60 + game_df['sec'].astype('int')
    game_df = game_df.reset_index(drop=True)
    start = game_df.time.iloc[0]
    end = game_df.time.iloc[-1]
    
    # ---player---
    raw = get_player_sample(game_df, start, end)
    data = raw.copy()
    X = player_column_trans.transform(data)
    pred_player = player_model.predict(X)

    # ---team---
    raw = get_team_sample(game_df, start, end)
    data = concat_home_and_away_team(raw)
    X = next_team_column_trans.transform(data)
    pred_next_team = next_team_model.predict(X)

    # ---xy---
    data = concat_home_and_away_xy(raw)
    X = next_x_column_trans.transform(data)
    pred_next_x = next_x_model.predict(X)
    X = next_y_column_trans.transform(data)
    pred_next_y = next_y_model.predict(X)

    result = [pred_player[0], pred_next_team[0], pred_next_y[0], pred_next_x[0]]
    result = ','.join([str(x) for x in result])
    with open('res_psgx.csv', 'w') as f:
        f.write(result)



# xml = etree.parse(GAME_FILE)
# Result(xml)
