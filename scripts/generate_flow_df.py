import os
import pandas as pd


print("started processing")
df = pd.read_hdf('data/pems-bay.h5')
sensor_ids = df.columns.values

for filename in sorted(os.listdir('data/pems-bay-raw')):
    print('processing:', filename)
    if filename == 'd04_text_station_5min_2017_06_01.txt':
        print('finished processing')
        break
    with open('data/pems-bay-raw/' + filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            data = line.split(',')
            time = pd.to_datetime(data[0])
            sensor_id = int(data[1])
            flow = float(data[9]) if data[9] != '' else 0
            if sensor_id in sensor_ids:
                df.at[time, sensor_id] = flow

print('saving new dataframe')
df.to_hdf('data/pems-bay-flow.h5')
