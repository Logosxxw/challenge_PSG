# challenge_PSG

#### challenge link: [Sports Analytics Challenge](https://www.agorize.com/zh/challenges/xpsg)

#### *run all scripts in the path:  challenge_PSG/code*

#### packages used:
pandas==0.23.4 
numpy==1.15.4 
lxml==4.3.3
scikit-learn==0.20.3 
xgboost==0.82 
tensorflow==1.12.0
absl-py==0.7.0

#### 1.transform original game xml to user friendly game xml with event type and qualifier type

```bash
for x in $(ls ../resources/raw_games/):
do
echo $x
python3 transform_xml --dim_player_team '../resources/Players and IDs - F40 - L1 20162017.xml' \
--dim_event '../resources/dim/event.txt' \
--dim_qualifier '../resources/dim/qualifier.txt' \
--game_file $x\
--output_dir '../resources/games_friendly'
done
```

#### 2.Statistics player playing time and get player information

```bash
python3 players.py --game_dir '../resources/games_friendly/' \
--player_file '../resources/Players and IDs - F40 - L1 20162017.xml' \
--output_use_player '../resources/dim/use_player.pkl' \
--output_player_info '../resources/dim/players.tsv'
```

#### 3.1 make sample by loop through all 15 minutes from a game and calculate team's performance (shoot, pass, foul, and so on...) as team features.

```bash
python3 make_sample_team.py --event_file '../resources/dim/event.txt' \
--qualifier_file '../resources/dim/qualifier.txt' \
--associate_file '../resources/dim/associate_use.tsv' \
--game_file '../resources/games_friendly/8_PSG_vs_Bordeaux.xml' \
--save_dir '../resources/samples/team'
```

#### 3.2 make sample by loop through all 15 minutes from a game and calculate player's performance (shoot, pass, foul, and so on...) as player features.

```bash
python3 make_sample_player.py --event_file '../resources/dim/event.txt' \
--qualifier_file '../resources/dim/qualifier.txt' \
--associate_file '../resources/dim/associate_use.tsv' \
--game_file '../resources/games_friendly/8_PSG_vs_Bordeaux.xml' \
--save_dir '../resources/samples/player' \
--player_file '../resources/dim/players.tsv' \
--use_player_file '../resources/dim/use_player'
```

#### 3.3 run the program in parallel for speed up

```bash
for x in {1..19}
do
    echo $x
    bash bash_scripts/$x.sh &
done
```

#### 4.training

I take first 18 rounds of games as training set and the 19th round as the valid set. 

The model scoring below is evaluation on the valid set based on model built from training set.

#### 4.1 training player model (gbdt)

accuracy on valid set: 9%

```bash
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
```

#### 4.2 training next event team model (gbdt)

accuracy on valid set: 78%

```bash
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
```

#### 4.3 training next event team model (rnn)

accuracy on valid set: 82% (improving 4% compared to gbdt)

```bash
python3 train_next_team_rnn.py --sample_dir '../resources/samples/team' \
--model_dir 'model' \
--dim_game '../resources/dim/games.csv' \
--tokenizer_file 'last10_tokenizer.pkl' \
--model_json 'next_team_rnn.json' \
--weights_file 'next_team_rnn.hdf5' \
--patience 5
```

#### 4.4 training next event x model (gbdt)

mae on valid set 16.80

```bash
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
```

#### 4.5 training next event y model (gbdt)

mae on valid set 24.66

```bash
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
```
