import pandas as pd

# 确定事件分类
associate = pd.read_csv('resources/dim/associate.tsv', sep='\t')
event_df = pd.read_csv('resources/dim/event.txt', sep='|')
qualifier_df = pd.read_csv('resources/dim/qualifier.txt', sep='|')

with open('resources/event_associate.txt', 'w') as f:
    for i in range(len(associate)):
        f.write(event_df.loc[event_df.id == associate.Type_id[i], ['id', 'type']].to_string())
        f.write("\n------------------------------------\n")
        associate_df = pd.DataFrame(data={'id':[int(x) for x in associate.qualifier_id[i].split(',')]})
        f.write(pd.merge(qualifier_df.loc[:,['id','type']], associate_df, on='id').to_string())
        f.write("\n====================================\n\n")
