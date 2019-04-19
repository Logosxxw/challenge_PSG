import pandas as pd
from logos_tools import *

UNK='null'
DEGREE_PER_RADIANS = 57.2957795
SECONDS_15MINUTES = 15*60

def pass_length_class(event):
    q = event.xpath("child::Q[@qualifier_id='212']")
    if len(q)>0:
        try:
            l = float(q[0].get('value'))
        except:
            return UNK
        if l<12:
            return 'short'
        elif l<32:
            return 'middle'
        else:
            return 'long'
    else:
        return UNK

def pass_direction_class(event):
    q = event.xpath("child::Q[@qualifier_id='213']")
    if len(q)>0:
        try:
            d = float(q[0].get('value')) * DEGREE_PER_RADIANS
        except:
            return UNK
        if d>=45 and d<135:
            return 'left'
        elif d>=135 and d<225:
            return 'back'
        elif d>=225 and d<315:
            return 'right'
        else:
            return 'forward'
    else:
        return UNK

def postion_class(event):
    q = event.xpath("child::Q[@qualifier_id='56']")
    if len(q)>0:
        return q[0].get('value', UNK)
    else:
        return UNK


def handle_special_q_type_player(event_id, q_type):
    if (q_type in ['Small boxcentre', 'Box-centre', 'Small box-right', 'Small box-left', 'Box-right', 'Box-left']):
        return 'inside box'
    elif (q_type in ['Out of boxcentre', 'Box-deep right', 'Box-deep left', 'Out of box-right', 'Out of box-left', 'Out of box-deep right', 'Out of box-deep left']):
        return 'out of box'
    elif (q_type in ['35+ centre', '35+ right', '35+ left']):
        return '35+'
    else:
        return q_type

def handle_special_q_type_team(event_id, q_type):
    q_type = handle_special_q_type_player(event_id, q_type)
    if (event_id in ['13', '14', '15', '16']):
        if (q_type in ['Right footed', 'Left footed']):
            return 'footed'
        elif (q_type in ['Volley', 'Individual Play']):
            return None
    elif (event_id == '10'):
        if (q_type in ['Reaching', 'Hands', 'Feet']):
            return None
    return q_type

def qualifier_class(event, associate_dict, qualifier_dict, handle_q_type):
    event_id = event.get('type_id')
    use_q_ids = associate_dict.get(event_id)
    if len(event)>0:
        q_list = []
        for x in event:
            q_id = x.get('qualifier_id')
            if (use_q_ids is not None):
                if(q_id in use_q_ids):
                    q_type = handle_q_type(event_id, qualifier_dict.get(q_id, UNK))
                    if (q_type is not None):
                        q_list.append(q_type)
        if(len(q_list)>0):
            q_list.sort()
            return '|'.join(q_list)
    return UNK

# 处理xml获取dataframe
def parse_event(event, event_dict, qualifier_dict, associate_dict):
    # 获取event基础信息
    period = event.get('period_id')
    minute = event.get('min')
    sec = event.get('sec')
    x = event.get('x')
    y = event.get('y')
    player_id = event.get('player_id')
    team_id = event.get('team_id')
    # 获取事件类型
    event_type = event_dict.get(event.get('type_id'), UNK)
    # 获取outcome
    outcome = event.get('outcome', UNK)
    # 获取传球的长度分类
    length = pass_length_class(event)
    # 获取传球的角度分类
    direction = pass_direction_class(event)
    # 获取处理球的位置
    position = postion_class(event)
    # 获取事件附加说明
    qualifier = qualifier_class(event, associate_dict, qualifier_dict, handle_special_q_type_team)
    return pd.DataFrame({'period':[period], 'min':minute, 'sec':sec, 'x':x, 'y':y, 'player_id':player_id, 'team_id':team_id, 'event_type':event_type, 'outcome':outcome, 'length':length, 'direction':direction, 'position':position, 'qualifier':qualifier})

# 时段
def get_period(use_half, start):
    if (use_half=='1'):
        if (start <= 17.5*60):
            return '1'
        else:
            return '2'
    else:
        if (start<=62.5*60):
            return '3'
        else:
            return '4'

# 计算有效控球时间时筛选有用的event
def calculate_control_time_useful_df(df, is_tail):
    event_type_list = list(df.event_type)
    result = None
    if ('Pass' in event_type_list):
        result = df.loc[df.event_type=='Pass']
    elif ('Take On' in event_type_list):
        result = df.loc[(df.event_type=='Take On') & (df.outcome=='1')]
    if (is_tail):
        if ('Foul' in event_type_list):
            result = df.loc[(df.event_type=='Foul') & (df.outcome=='1')]
        elif ('Miss' in event_type_list):
            result = df.loc[df.event_type=='Miss']
        elif ('Post' in event_type_list):
            result = df.loc[df.event_type=='Post']
        elif ('Attempt Saved' in event_type_list):
            result = df.loc[df.event_type=='Attempt Saved']
        elif ('Goal' in event_type_list):
            result = df.loc[df.event_type=='Goal']
    if (result is not None and len(result)>0):
        return result.iloc[0]
    else:
        return None

# 控球时长统计
def calculate_control_time(use_time_unique, use_df):
    control_time_dict = {'0':0, '1':0}
    for i in range(len(use_time_unique)-1):
        pre_df = use_df.loc[use_df.time==use_time_unique.iloc[i]]
        df = use_df.loc[use_df.time==use_time_unique.iloc[i+1]]
        pre_df = calculate_control_time_useful_df(pre_df, False)
        df = calculate_control_time_useful_df(df, True)
        if (pre_df is not None and df is not None):
            if (pre_df.team_id == df.team_id):
                control_time_dict[pre_df.team_id] += (int(df.time)-int(pre_df.time))
    return pd.DataFrame(pd.Series(control_time_dict), columns=['control_time'])

def calculate_shoot_data(df, s, total):
    ct = len(df.loc[s])
    rate = handle_divided_by0(ct, total)
    return (ct, rate)

def shoot_stat(df):
    shoot_ct = len(df)
    onTarget = calculate_shoot_data(df, (df.event_type.isin(['Post', 'Attempt Saved', 'Goal'])) & (~df.qualifier.str.contains('Blocked')), shoot_ct)
    bigChance = calculate_shoot_data(df, df.qualifier.str.contains('Big Chance'), shoot_ct)
    inside = calculate_shoot_data(df, df.qualifier.str.contains('inside box'), shoot_ct)
    outside = calculate_shoot_data(df, df.qualifier.str.contains('out of box'), shoot_ct)
    head = calculate_shoot_data(df, df.qualifier.str.contains('Head'), shoot_ct)
    return pd.DataFrame({'shoot':[shoot_ct], 
                  'onTarget':onTarget[0], 'onTarget_rate':onTarget[1],
                  'bigChance':bigChance[0], 'bigChance_rate':bigChance[1], 
                  'inside':inside[0], 'inside_rate':inside[1],
                  'outside':outside[0], 'outside_rate':outside[1],
                  'head':head[0], 'head_rate':head[1]})

def calculate_pass_data(df, s, total):
    ct = len(df.loc[s])
    success = len(df.loc[(s) & (df.outcome=='1')])
    rate = handle_divided_by0(ct, total)
    success_rate = handle_divided_by0(success, ct)
    return (ct, success, rate, success_rate)

def pass_stat(df):
    # 传球
    pass_ct = len(df)
    pass1_rate = len(df.loc[df.outcome=='1'])/pass_ct
    # 前场传球
    front = calculate_pass_data(df, df.position.isin(['Center','Right', 'Left']), pass_ct)
    # 关键传球
    key = calculate_pass_data(df, df.qualifier.str.contains('Assist'), front[0])
    # 传中
    cross = calculate_pass_data(df, df.qualifier.str.contains('Cross'), front[0])
    # 直塞
    through = calculate_pass_data(df, df.qualifier.str.contains('Through ball'), front[0])
    # 高球
    chipped = calculate_pass_data(df, df.qualifier.str.contains('Chipped'), pass_ct)
    # 前后左右
    forward = calculate_pass_data(df, df.direction=='forward', pass_ct)
    back = calculate_pass_data(df, df.direction=='back', pass_ct)
    left = calculate_pass_data(df, df.direction=='left', pass_ct)
    right = calculate_pass_data(df, df.direction=='right', pass_ct)
    # 距离
    short = calculate_pass_data(df, df.length=='short', pass_ct)
    middle = calculate_pass_data(df, df.length=='middle', pass_ct)
    long = calculate_pass_data(df, df.length=='long', pass_ct)
    # 位置
    positionC = calculate_pass_data(df, df.position=='Center', pass_ct)
    positionL = calculate_pass_data(df, df.position=='Left', pass_ct)
    positionR = calculate_pass_data(df, df.position=='Right', pass_ct)
    positionB = calculate_pass_data(df, df.position=='Back', pass_ct)
    return pd.DataFrame({'pass':[pass_ct], 'pass1_rate':pass1_rate, 
                         'front':front[0], 'front_rate':front[2], 'front1_rate':front[3],
                         'key':key[0], 'key_rate':key[2],
                         'cross':cross[0], 'cross_rate':cross[2], 'cross1_rate':cross[3],
                         'through':through[0], 'through_rate':through[2], 'through1_rate':through[3],
                         'chipped':chipped[0], 'chipped_rate':chipped[2], 'chipped1_rate':chipped[3],
                         'forward':forward[0], 'forward_rate':forward[2], 'forward1_rate':forward[3],
                         'back':back[0], 'back_rate':back[2], 'back1_rate':back[3],
                         'left':left[0], 'left_rate':left[2], 'left1_rate':left[3],
                         'right':right[0], 'right_rate':right[2], 'right1_rate':right[3],
                         'short':short[0], 'short_rate':short[2], 'short1_rate':short[3],
                         'middle':middle[0], 'middle_rate':middle[2], 'middle1_rate':middle[3],
                         'long':long[0], 'long_rate':long[2], 'long1_rate':long[3],
                         'positionC':positionC[0], 'positionC_rate':positionC[2], 'positionC1_rate':positionC[3],
                         'positionL':positionL[0], 'positionL_rate':positionL[2], 'positionL1_rate':positionL[3],
                         'positionR':positionR[0], 'positionR_rate':positionR[2], 'positionR1_rate':positionR[3],
                         'positionB':positionB[0], 'positionB_rate':positionB[2], 'positionB1_rate':positionB[3]
                        })

def calculate_outcome_data(df, s, outcome='1'):
    ct = len(df.loc[s])
    success = len(df.loc[(s) & (df.outcome==outcome)])
    success_rate = handle_divided_by0(success, ct)
    return (ct, success_rate)

def other_stat(df):
    # 角球
    corner = len(df.loc[df.qualifier.str.contains('Corner taken')])
    # 任意球
    freeKick = len(df.loc[df.qualifier.str.contains('Free kick')])
    # 越位
    offside = len(df.loc[df.event_type=='Offside'])
    # 封堵射门
    defBlock = len(df.loc[df.qualifier.str.contains('Def block')])
    # 抢断
    tackle = calculate_outcome_data(df, df.event_type=='Tackle')
    # 拦截
    interception = len(df.loc[df.event_type=='Interception'])
    # 过人
    takeon = calculate_outcome_data(df, df.event_type=='Take On')
    # 被过
    challenge = len(df.loc[df.event_type=='Challenge'])
    # 糟糕的触球
    ballTouch = calculate_outcome_data(df, df.event_type=='Ball touch', '0')
    # 解围
    clearance = len(df.loc[df.event_type=='Clearance'])
    # 扑救
    save = len(df.loc[df.event_type=='Save'])
    # 犯规
    foul = len(df.loc[(df.event_type=='Foul') & (df.outcome=='1')])
    fouled = len(df.loc[(df.event_type=='Foul') & (df.outcome=='0')])
    # 黄牌
    yellow = len(df.loc[(df.event_type=='Card')&(df.qualifier.str.contains('yellow',case=False))])
    # 红牌
    red = len(df.loc[(df.event_type=='Card')&(df.qualifier.str.contains('Red card'))])
    return pd.DataFrame({'corner':[corner], 'freeKick':freeKick, 'offside':offside, 'defBlock':defBlock, 
     'tackle':tackle[0], 'tackle1_rate':tackle[1], 'interception':interception, 
     'takeon':takeon[0], 'takeon1_rate':takeon[1], 'challenge':challenge, 
     'ballTouch':ballTouch[0], 'ballTouch0_rate':ballTouch[1], 'clearance':clearance,
     'save':save, 'foul':foul, 'fouled':fouled, 'yellow':yellow, 'red':red})

def get_last20_seq(df, use_period):
    last20 = df.iloc[-20:]
    last20_list = list(last20.event_type +'-'+ last20.position +'-'+ use_period)
    return ','.join(last20_list)

# 根据xy判断区域
def get_position_xy(x,y):
    try:
        x = float(x)
        y = float(y)
        if (x<34):
            if (y<21.1):
                return 'rightB'
            elif (y>=78.9):
                return 'leftB'
            elif (x<17):
                return 'boxB'
            else:
                return 'outofboxB'
        elif (x<66):
            return 'middle'
        else:
            if (y<21.1):
                return 'rightF'
            elif (y>=78.9):
                return 'leftF'
            elif (x>=83):
                return 'boxF'
            else:
                return 'outofboxF'
    except:
        return ''

# 削减样本防止过拟合
def sub_sample(df, step):
        unique_start = df.start.unique()
        return df.loc[df.start.isin(unique_start[list(range(0, len(unique_start), step))])]

def add_new_features(data):
    def get_event_use(x, i):
        return '-'.join(x.split(',')[-i].split('-')[:2])
    data['last1_event'] = data.last10_list.map(lambda x: get_event_use(x,1))
    data['last2_event'] = data.last10_list.map(lambda x: get_event_use(x,2))
    data['pass1'] = data['pass']*data['pass1_rate']
    data['front1'] = data['front']*data['front1_rate']
    data['defend'] = data['tackle']+data['interception']
    data['takeon1'] = data['takeon1_rate']+data['takeon']
    data['positionC1'] = data['positionC1_rate']*data['positionC']
    data['positionL1'] = data['positionL1_rate']*data['positionL']
    data['positionR1'] = data['positionR1_rate']*data['positionR']
    data['positionB1'] = data['positionB1_rate']*data['positionB']
    data['last_position'] = data.last_event.map(lambda x:x.split('-')[2])
    data['position9'] = data.last10_list.map(lambda x:x.split(',')[9].split('-')[2])
    data['position8'] = data.last10_list.map(lambda x:x.split(',')[8].split('-')[2])
    data['position7'] = data.last10_list.map(lambda x:x.split(',')[7].split('-')[2])
    data['position6'] = data.last10_list.map(lambda x:x.split(',')[6].split('-')[2])
    data['position5'] = data.last10_list.map(lambda x:x.split(',')[5].split('-')[2])
    return data


def get_train_test_index_from_rounds(dim_game, games_s, test_rounds):
    games_df = pd.read_csv(dim_game)
    test_games_df = games_df.loc[games_df.rounds.isin(test_rounds)]
    test_games = test_games_df.rounds.astype('str')+'_'+test_games_df.homes+'_vs_'+test_games_df.aways
    return (list(games_s[~games_s.isin(test_games)].index), list(games_s[games_s.isin(test_games)].index))