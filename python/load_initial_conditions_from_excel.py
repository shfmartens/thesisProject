import pandas as pd
import json
from collections import OrderedDict

# Merge conditions

with open('../config/config.json') as data_file:
    config = json.load(data_file)

df1 = pd.read_excel('near_vertical.xlsx', header=None).rename(columns={0: 'x', 1: 'y_dot', 2: 'z_dot'})
df2 = pd.DataFrame.from_dict(config['near_vertical']).T.reset_index(drop=True).filter(['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot'])

df = pd.concat([df1, df2]).fillna(0).sort_values('x').reset_index(drop=True)
df['new_index'] = 0
for i in range(len(df)):
    df['new_index'][i] = 'near_vertical_' + str(i+1)

df['z_dot'] = abs(df['z_dot'])
unordered_dict = df.set_index('new_index', drop=True).T.to_dict()
