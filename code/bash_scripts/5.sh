for x in $(ls ../resources/games_friendly/5_*)
do 
echo "${x}"
python3 make_sample_team.py --event_file '../resources/dim/event.txt' --qualifier_file '../resources/dim/qualifier.txt' --associate_file '../resources/dim/associate_use.tsv' --game_file "${x}" --save_dir '../resources/samples/team'
done