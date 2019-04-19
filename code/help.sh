# 1.transform original game xml to user friendly game xml with event type and qualifier type
for x in $(ls ../resources/raw_games/):
do
echo $x
python3 transform_xml --dim_player_team '../resources/Players and IDs - F40 - L1 20162017.xml' \
--dim_event '../resources/dim/event.txt' \
--dim_qualifier '../resources/dim/qualifier.txt' \
--game_file $x\
--output_dir '../resources/games_friendly'
done

# 2.Statistics player playing time and get player information
python3 players.py --game_dir '../resources/games_friendly/' \
--player_file '../resources/Players and IDs - F40 - L1 20162017.xml' \
--output_use_player '../resources/dim/use_player.pkl' \
--output_player_info '../resources/dim/players.tsv'

# 3.make sample by loop through all 15 minutes from a game and calculate team's performance (shoot, pass, foul, and so on...) as team features.
python3 make_sample_team.py --event_file '../resources/dim/event.txt' \
--qualifier_file '../resources/dim/qualifier.txt' \
--associate_file '../resources/dim/associate_use.tsv' \
--game_file '../resources/games_friendly/8_PSG_vs_Bordeaux.xml' \
--save_dir '../resources/samples/team'

# make sample by loop through all 15 minutes from a game and calculate player's performance (shoot, pass, foul, and so on...) as player features.
python3 make_sample_player.py --event_file '../resources/dim/event.txt' \
--qualifier_file '../resources/dim/qualifier.txt' \
--associate_file '../resources/dim/associate_use.tsv' \
--game_file '../resources/games_friendly/8_PSG_vs_Bordeaux.xml' \
--save_dir '../resources/samples/player' \
--player_file '../resources/dim/players.tsv' \
--use_player_file '../resources/dim/use_player'

# run the program in parallel for speed up
for x in {1..19}
do
    echo $x
    bash bash_scripts/$x.sh &
done

# training player model (gbdt)
python3 train_player_model.py --sample_dir '../resources/samples/player' \
--model_dir 'model' \
--dim_game '../resources/dim/games.csv' \
--transformer_file 'player_transformer.pkl' \
--model_file 'player_model.pkl' \
--trees 2000 \
--patience 100 \
--xgb_n_jobs 20 \
--depth_search '3' \
--gamma_search '0.5,2' \
--alpha_search '0.03125'

# training next event team model (gbdt)
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

# training next event team model (rnn)
python3 train_next_team_rnn.py --sample_dir '../resources/samples/team' \
--model_dir 'model' \
--dim_game '../resources/dim/games.csv' \
--tokenizer_file 'last10_tokenizer.pkl' \
--model_json 'next_team_rnn.json' \
--weights_file 'next_team_rnn.hdf5' \
--patience 5

# training next event x model (gbdt)
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

# training next event y model (gbdt)
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
