import pandas as pd
from lxml import etree
from absl import flags, app
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('dim_player_team', None, '')
flags.DEFINE_string('dim_event', None, '')
flags.DEFINE_string('dim_qualifier', None, '')
flags.DEFINE_string('game_file', None, '')
flags.DEFINE_string('output_dir', None, '')

DEGREE_PER_RADIANS = 57.2957795

def main(_):
    # 将球员球队读取为dict
    xml = etree.parse(FLAGS.dim_player_team)
    team_id = []
    team_name = []
    player_id = []
    player_name = []
    position = []
    real_position = []
    real_position_side = []
    jersey_num = []
    for element in xml.iter(tag=etree.Element):
        if (element.tag=='Team'):
            team_id.append(element.get('uID')[1:])
            team_name.append(element.xpath('Name')[0].text)
        if (element.tag=='Player'):
            player_id.append(element.get('uID')[1:])
            player_name.append(element.xpath('Name')[0].text)
            position.append(element.xpath('Position')[0].text)
            real_position.append(element.xpath("Stat[@Type='real_position']")[0].text)
            real_position_side.append(element.xpath("Stat[@Type='real_position_side']")[0].text)
            jersey_num.append(element.xpath("Stat[@Type='jersey_num']")[0].text)
    team = dict()
    player = dict()
    rm_Unknown = lambda x : '' if x == 'Unknown' else x
    for i in range(len(team_id)):
        team[team_id[i]] = team_name[i]
    for i in range(len(player_id)):
        player[player_id[i]] = '%s, %s, %s, %s, %s' % (rm_Unknown(player_name[i]), rm_Unknown(jersey_num[i]), rm_Unknown(position[i]), rm_Unknown(real_position[i]), rm_Unknown(real_position_side[i]))

    # 加载维表
    event_df = pd.read_csv(FLAGS.dim_event, sep='|')
    event_s = pd.Series(data=event_df.type.values, index=[str(x) for x in event_df.id])
    event = event_s.to_dict()
    qualifier_df = pd.read_csv(FLAGS.dim_qualifier, sep='|')
    qualifier_s = pd.Series(data=qualifier_df.type.values, index=[str(x) for x in qualifier_df.id])
    qualifier = qualifier_s.to_dict()

    # 读取比赛
    xml = etree.parse(FLAGS.game_file)
    def delete_element_attributes(element, l):
        for x in l:
            if element.get(x) is not None: del element.attrib[x]

    # 遍历比赛事件并关联维表
    game_name = ''
    for element in xml.iter(tag=etree.Element):
        if (element.tag=='Game'):
            game_name = '%s_%s_vs_%s' % (element.get('matchday'), element.get('home_team_name'), element.get('away_team_name'))
        if (element.tag=='Event'):
            element.set('type', event.get(element.get('type_id'), ''))
            delete_element_attributes(element, ['id', 'timestamp', 'last_modified', 'version'])
            if element.get('team_id') is not None: element.set('team', team.get(element.get('team_id'), ''))
            if element.get('player_id') is not None: element.set('player', player.get(element.get('player_id'), ''))
        if (element.tag=='Q'):
            qid = element.get('qualifier_id')
            element.set('type', qualifier.get(qid, ''))
            delete_element_attributes(element, ['id'])
            if qid=='213':
                element.set('degree', str(round(float(element.get('value')) * DEGREE_PER_RADIANS, 0)))

    # 保存新的xml文件
    with open(os.path.join(FLAGS.output_dir, game_name+'.xml'), 'w') as f:
        f.write(etree.tostring(xml, encoding='unicode', method='xml', pretty_print=True))

if __name__ == '__main__':
    app.run(main)
