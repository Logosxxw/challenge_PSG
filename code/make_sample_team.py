import pandas as pd
import numpy as np
import pickle
import os
from lxml import etree
from absl import flags, logging, app
from logos_tools import *
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

    # 制作样本
    result1 = make_half_sample('1', game_df, False)
    result2 = make_half_sample('2', game_df, False)
    result = pd.concat([result1, result2], axis=0, sort=False)
    result = result.reset_index(drop=True)
    team_info_df = pd.DataFrame({'team_id':['0', '1'], 'team_id_real':[away_team_id, home_team_id], 'team_name':[away_team_name, home_team_name]})
    result = pd.merge(result, team_info_df, on='team_id')
    save_name = FLAGS.game_file.split('/')[-1].split('.')[0] + '.tsv'
    result.to_csv(os.path.join(FLAGS.save_dir, save_name), sep='\t', index=False)


if __name__ == '__main__':
    app.run(main)
