# 0.安装python包

# 1.处理原始的game xml，关联event与qualifier维表，生成更易于数据探索xml文件
for x in $(ls ../resources/raw_games/):
do
echo $x
python3 transform_xml --dim_player_team '../resources/Players and IDs - F40 - L1 20162017.xml' \
--dim_event '../resources/dim/event.txt' \
--dim_qualifier '../resources/dim/qualifier.txt' \
--game_file $x\
--output_dir '../resources/games_friendly'
done

# 2.处理球员数据，得到800分钟+球员以及球员维表
python3 players.py --game_dir '../resources/games_friendly/' \
--player_file '../resources/Players and IDs - F40 - L1 20162017.xml' \
--output_use_player '../resources/dim/use_player.pkl' \
--output_player_info '../resources/dim/players.tsv'

# 3.解析比赛构造球员球队样本，以15分钟为单位，每次向前滑动一个时间点生成一个样本
python3 make_sample_team.py --event_file '../resources/dim/event.txt' \
--qualifier_file '../resources/dim/qualifier.txt' \
--associate_file '../resources/dim/associate_use.tsv' \
--game_file '../resources/games_friendly/8_PSG_vs_Bordeaux.xml' \
--save_dir '../resources/samples/team'

python3 make_sample_player.py --event_file '../resources/dim/event.txt' \
--qualifier_file '../resources/dim/qualifier.txt' \
--associate_file '../resources/dim/associate_use.tsv' \
--game_file '../resources/games_friendly/8_PSG_vs_Bordeaux.xml' \
--save_dir '../resources/samples/player' \
--player_file '../resources/dim/players.tsv' \
--use_player_file '../resources/dim/use_player'

# 并行构造样本，提升速度
for x in {1..19}
do
    echo $x
    bash bash_scripts/$x.sh &
done

# 训练球员模型
python3 train_player_model.py --sample_dir '../resources/samples/player' \
--model_dir 'model' \
--dim_game '../resources/dim/games.csv' \
--transformer_file 'player_transformer.pkl' \
--model_file 'player_model.pkl' \
--trees 2000 \
--patience 100 \
--xgb_n_jobs 20 \
--depth_search '3' \
--gamma_search '0.5,0' \
--alpha_search '0.5,0'

# 训练next event球队模型
python3 train_next_team_model.py --sample_dir '../resources/samples/team' \
--model_dir 'model' \
--dim_game '../resources/dim/games.csv' \
--transformer_file 'next_team_transformer.pkl' \
--model_file 'next_team_model.pkl' \
--trees 2000 \
--patience 100 \
--xgb_n_jobs 20 \
--depth_search '3' \
--gamma_search '0.5,0' \
--alpha_search '0.5,0'

# 训练next event xy模型
python3 train_next_xy_model.py --sample_dir '../resources/samples/team' \
--model_dir 'model' \
--dim_game '../resources/dim/games.csv' \
--transformer_file 'next_x_transformer.pkl' \
--model_file 'next_x_model.pkl' \
--trees 2000 \
--patience 100 \
--xgb_n_jobs 20 \
--depth_search '3' \
--gamma_search '0.5,0' \
--alpha_search '0.5,0' \
--predict_y=False

python3 train_next_xy_model.py --sample_dir '../resources/samples/team' \
--model_dir 'model' \
--dim_game '../resources/dim/games.csv' \
--transformer_file 'next_y_transformer.pkl' \
--model_file 'next_y_model.pkl' \
--trees 2000 \
--patience 100 \
--xgb_n_jobs 20 \
--depth_search '3' \
--gamma_search '0.5,0' \
--alpha_search '0.5,0' \
--predict_y=True
