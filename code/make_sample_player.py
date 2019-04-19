import pandas as pd
import numpy as np
import pickle
import os
from lxml import etree
from absl import flags, logging, app
from logos_opta import *

# EVENT_FILE = '../resources/dim/event.txt'
# QUALIFIER_FILE = '../resources/dim/qualifier.txt'
# ASSOCIATE_FILE = '../resources/dim/associate_use.tsv'
# GAME_FILE = '../resources/games_friendly/17_Paris Saint-Germain_vs_Nice.xml'
# SAVE_DIR = '../resources/samples/team'
FLAGS = flags.FLAGS
flags.DEFINE_string('event_file', None, 'event_file')
flags.DEFINE_string('qualifier_file', None, 'qualifier_file')
flags.DEFINE_string('associate_file', None, 'associate_file')
flags.DEFINE_string('game_file', None, 'game_file')
flags.DEFINE_string('save_dir', None, 'save_dir')
flags.DEFINE_string('player_file', None, 'player_file')
flags.DEFINE_string('use_player_file', None, 'use_player_file')

# make one player same within 15 minutes
def make_one_sample_player(use_half, start, end, use_time_unique, use_df, use_players_df, use_control=False):
    use_players_ct = len(use_players_df)
    # period
    use_period = get_period(use_half, start)
    # control ball time
    control_df = calculate_control_time(use_time_unique, use_df) if use_control else None # not use because of the high calculation cost
    # shoot
    shoot_df = pd.DataFrame({'bigChance':[0]*use_players_ct, 'bigChance_rate':[0.0], 
    'head':[0], 'head_rate':[0.0], 
    'inside':[0], 'inside_rate':[0.0], 
    'onTarget':[0], 'onTarget_rate':[0.0], 
    'outside':[0], 'outside_rate':[0.0], 
    'shoot':[0]}, index=use_players_df.index)
    shoot_df.index.name = 'player_id'
    selected = use_df.loc[use_df.event_type.isin(['Miss', 'Post', 'Attempt Saved', 'Goal'])]
    if len(selected)>0:
        shoot_df = selected.groupby('player_id').apply(shoot_stat)
        shoot_df = shoot_df.reset_index(level=1, drop=True)
    # pass
    pass_df = pd.DataFrame({'pass':[0]*use_players_ct, 'pass1_rate':[0.0], 
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
                    }, index=use_players_df.index)
    pass_df.index.name = 'player_id'
    selected = use_df.loc[use_df.event_type=='Pass']
    if (len(selected)>0):
        pass_df = selected.groupby('player_id').apply(pass_stat)
        pass_df = pass_df.reset_index(level=1, drop=True)
    # other
    other_df = use_df.groupby('player_id').apply(other_stat)
    other_df = other_df.reset_index(level=1, drop=True)
    # last 20 events
    last20_s = use_df.groupby('player_id').apply(get_last20_seq, use_period=use_period)
    last20_s = last20_s.rename('last20_list')
    # team_id
    team_id_s = use_df.groupby('player_id').apply(lambda df:df.team_id.iloc[0])
    team_id_s = team_id_s.rename('team_id')
    # y
    y_df = use_players_df.loc[:,['player_id', 'player_name', 'position_use']]
    # concat
    sample = pd.concat([pass_df, shoot_df, other_df, control_df, last20_s, team_id_s, y_df], axis=1, sort=False)
    sample = sample.fillna(0)
    sample['period'] = use_period
    sample['start'] = start
    sample['end'] = end
    return sample

# for whole half time samples
def make_half_sample_player(use_half, game_df, use_players_all, all_players_df):
    half_df = game_df.loc[game_df.period==use_half]
    time_unique_s = half_df.groupby('time').apply(lambda df: list(df.index))
    time_unique = pd.Series(time_unique_s.index)
    # 限定可以被循环的开始时间（在半场内至少能满15分钟）
    end_max = time_unique.iloc[-2]
    start_max = end_max - SECONDS_15MINUTES
    result = []
    time_unique_loop = time_unique.loc[time_unique<=start_max]
    ct = len(time_unique_loop)
    # 球员触球机会不密集，每隔10个事件采集一次
    for i in range(0, ct, 10):
        # print(round(i*1.0/ct, 4))
        start = time_unique_loop.iloc[i]
        print(start)
        end = time_unique[time_unique <= (start+SECONDS_15MINUTES)].iloc[-1]
        use_df = half_df.loc[(half_df.time>=start) & (half_df.time<=end)]
        # 最后十个事件不完整，不纳入参考
        use_df = use_df.iloc[:-10]
        end = use_df.time.iloc[-1]
        use_time_unique = time_unique.loc[(time_unique>=start) & (time_unique<=end)]
        # 获取时段内的所有球员
        all_players = set(use_df.player_id.unique())
        # 关联需要考察的球员
        use_players = all_players.intersection(use_players_all)
        # 进一步缩减use_df
        use_df = use_df.loc[use_df.player_id.isin(use_players)]
        use_players_df = all_players_df.loc[use_players]
        sample = make_one_sample_player(use_half, start, end, use_time_unique, use_df, use_players_df)
        # result.append(sample)
        try:
            sample = make_one_sample_player(use_half, start, end, use_time_unique, use_df, use_players_df)
            result.append(sample)
        except:
            print('make_one_sample_player error')
    return pd.concat(result, axis=0, sort=False)

def main(_):
    # print(FLAGS.event_file, FLAGS.qualifier_file, FLAGS.associate_file, FLAGS.game_file, FLAGS.save_dir)
    # 加载维表
    event_df = pd.read_csv(FLAGS.event_file, sep='|')
    event_s = pd.Series(data=event_df.type.values, index=[str(x) for x in event_df.id])
    event_dict = event_s.to_dict()
    qualifier_df = pd.read_csv(FLAGS.qualifier_file, sep='|')
    qualifier_s = pd.Series(data=qualifier_df.type.values, index=[str(x) for x in qualifier_df.id])
    qualifier_dict = qualifier_s.to_dict()
    associate_df = pd.read_csv(FLAGS.associate_file, sep='\t')
    associate_dict = {str(associate_df.Type_id[i]):associate_df.qualifier_id[i].split(',') for i in range(len(associate_df))}

    # 读取比赛
    xml = etree.parse(FLAGS.game_file)

    # 获取主客队信息
    game = xml.xpath('Game')[0]
    away_team_id = game.get('away_team_id')
    away_team_name = game.get('away_team_name')
    home_team_id = game.get('home_team_id')
    home_team_name = game.get('home_team_name')

    # 处理事件，将xml转化为dataframe
    game_df = pd.concat([parse_event(x, event_dict, qualifier_dict, associate_dict) for x in game], axis=0)
    game_df = game_df.fillna(value=UNK)
    # Deleted event去除附加信息
    game_df.loc[game_df.event_type=='Deleted event', 
                ['length', 'direction', 'position', 'qualifier']] = UNK
    # Clearance去除方向与位置信息
    game_df.loc[game_df.event_type=='Clearance', 
                ['direction', 'position']] = UNK
    # 去除Start/END事件
    game_df = game_df.loc[~game_df.event_type.isin(['Team set up', 'Start', 'End', 'Collection End'])]
    # 标注主客队
    game_df['team_id_real'] = game_df['team_id']
    game_df['team_id'] = '1'
    game_df.loc[game_df.team_id_real==away_team_id, 'team_id'] = '0'
    # 便于统计的时间
    game_df['time'] = game_df['min'].astype('int')*60 + game_df['sec'].astype('int')
    # 重新整理index
    game_df.index = list(range(len(game_df)))

    # 加载800分钟球员
    with open(FLAGS.use_player_file, 'rb') as handle:
        use_players_all=pickle.load(handle)
    # 加载球员信息
    all_players_df = pd.read_csv(FLAGS.player_file, sep='\t')
    all_players_df.index = [str(x) for x in all_players_df.player_id]
    # 制作样本
    result1 = make_half_sample_player('1', game_df, use_players_all, all_players_df)
    result2 = make_half_sample_player('2', game_df, use_players_all, all_players_df)
    # 保存数据
    result = pd.concat([result1, result2], axis=0, sort=False)
    result = result.reset_index(drop=True)
    team_info_df = pd.DataFrame({'team_id':['0', '1'], 'team_id_real':[away_team_id, home_team_id], 'team_name':[away_team_name, home_team_name]})
    result = pd.merge(result, team_info_df, on='team_id')
    save_name = FLAGS.game_file.split('/')[-1].split('.')[0] + '.tsv'
    result.to_csv(os.path.join(FLAGS.save_dir, save_name), sep='\t', index=False)


if __name__ == '__main__':
    app.run(main)
