import os
import pandas as pd


print("started processing")
df = pd.read_hdf('data/pems-bay.h5')
sensor_ids = df.columns.values

raw_dir = 'data/pems-bay-raw'
frames = []
for filename in sorted(os.listdir(raw_dir)):
    print('processing:', filename)
    if filename == 'd04_text_station_5min_2017_07_01.txt':
        print('finished processing')
        break
    df = pd.read_csv(os.path.join(raw_dir, filename), sep=',', header=None)
    df = df.iloc[:, [0, 1, 9]]
    df.columns = ['time', 'sensor_id', 'flow']
    df = df[df['sensor_id'].isin(sensor_ids)]
    df['time'] = pd.to_datetime(df['time'])
    df.fillna(0, inplace=True)
    df = df.pivot_table(values='flow', index='time', columns='sensor_id')
    frames.append(df)

final_df = pd.concat(frames)
print('saving new dataframe')
final_df.to_hdf('data/pems-bay-flow.h5', key='df')
