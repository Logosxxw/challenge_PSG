import pandas as pd
import numpy as np
import pickle
import os
from lxml import etree
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string('game_dir', None, '')
flags.DEFINE_string('player_file', None, '')
flags.DEFINE_string('output_use_player', None, '')
flags.DEFINE_string('output_player_info', None, '')

# ------筛选需要预测的球员------
def get_time(minute, sec):
    return int(minute)*60 + int(sec)

def get_player_time_df(xml):
    # 获取首发
    set_up = xml.xpath('//Event[@type="Team set up"]')
    def get_from_setup(set_up, qid):
        return sum([[x.strip() for x in event.xpath('Q[@qualifier_id="%s"]' % qid)[0].get('value').split(',')] for event in set_up], [])
    position_list = get_from_setup(set_up, '44')
    player_list = get_from_setup(set_up, '30')
    player_df = pd.DataFrame({'player':player_list, 'position':position_list})
    # 结束时间
    total_time = max([get_time(event.get('min'), event.get('sec')) for event in xml.xpath('//Event[@type="End"]')])
    # 简化处理，简单考虑中场的伤停补时
    player_df.loc[player_df.position!='5', 'start'] = 0
    player_df.loc[:, 'end'] = total_time
    # 球员上场，获取开始时间
    player_on = xml.xpath('//Event[@type="Player on"]')
    player_on_list = [(event.get('player_id'), get_time(event.get('min'), event.get('sec'))) for event in player_on]
    for item in player_on_list:
        player_df.loc[player_df.player==item[0], 'start'] = item[1]
    # 球员下场，获取结束时间
    player_off = xml.xpath('//Event[@type="Player Off"]')
    player_off_list = [(event.get('player_id'), get_time(event.get('min'), event.get('sec'))) for event in player_off]
    for item in player_off_list:
        player_df.loc[player_df.player==item[0], 'end'] = item[1]
    player_df = player_df.loc[player_df.start.notna()].copy()
    player_df.loc[:,'time'] = (player_df.end - player_df.start)
    return player_df

# 获取所有的比赛并获得球员总时间
game_dir = FLAGS.game_dir
all_games = [f for f in os.listdir(game_dir) if os.path.isfile(os.path.join(game_dir, f))]
player_time_list = [get_player_time_df(etree.parse(os.path.join(game_dir,x))) for x in all_games]
player_time_df = pd.concat(player_time_list)
player_s = player_time_df.groupby('player')['time'].sum()

# 获取出场时间>800分钟的球员（忽略了中场的伤停补时）
player_s = player_s.loc[player_s>(800*60)]

# 转会
player_file = FLAGS.player_file
xml = etree.parse(player_file)
playerChanges = xml.xpath('//PlayerChanges')[0]
playerChanges = playerChanges.xpath('Team/Player')
playerid = []
leave_date = []
for x in playerChanges:
    playerid.append(x.get('uID')[1:])
    leave_date.append(x.xpath('Stat[@Type="leave_date"]')[0].text)
playerChanges_df = pd.DataFrame({'player':playerid, 'leave_date':leave_date})

# 去除被交易的球员（理解有误，多删除了一些可以使用的球员）
player_s = player_s.loc[list(set(list(player_s.index)) - set(list(playerChanges_df.player)))]

# 保存要预测的球员
output_use_player = FLAGS.output_use_player
with open(output_use_player, 'wb') as handle:
    pickle.dump(set(player_s.index), handle, protocol=pickle.HIGHEST_PROTOCOL)


# ------获取球员信息------
def decide_side(x):
    if x in ('Left','Right','Left/Right'):
        return 'side'
    elif x == 'Centre':
        return 'centre'
    else:
        return 'multi'

player_id = []
player_name = []
position = []
real_position = []
real_position_side = []
jersey_num = []
for element in xml.xpath('SoccerDocument/Team/Player'):
    player_id.append(element.get('uID')[1:])
    player_name.append(element.xpath('Name')[0].text)
    position.append(element.xpath('Position')[0].text)
    real_position.append(element.xpath("Stat[@Type='real_position']")[0].text)
    real_position_side.append(element.xpath("Stat[@Type='real_position_side']")[0].text)
    jersey_num.append(element.xpath("Stat[@Type='jersey_num']")[0].text)
all_player_df = pd.DataFrame({'player_id':player_id, 'player_name':player_name, 'position':position, 
                              'real_position':real_position, 'real_position_side':real_position_side, 
                              'jersey_num':jersey_num}, index=player_id)
all_player_df['position_use'] = all_player_df.position + '-' + all_player_df.real_position_side.map(decide_side)

# 去重并保存
all_player_df = all_player_df.drop_duplicates(subset='player_id', keep='last')
output_player_info = FLAGS.output_player_info
all_player_df.to_csv(output_player_info, sep='\t', index=False)
