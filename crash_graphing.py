import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
import time

data_file = 'NetScreenCounts.csv'
df = pd.read_csv(data_file)

# get only highway classes that have a certain number of occurrences in the data
df = df.groupby('HighwayClassCode').filter(lambda x: len(x) >=25)

count_col = 'TotalCrashes'

df['CrashMileYear'] = df[count_col] / df['SegmentLength'] / 5

crash_col = 'CrashMileYear'

# Loop over all values of hwy class and graph each of them AADT vs TotalCrashes
# Get all unique hwy classes available
hwy_classes = df.HighwayClassCode.unique()

for hwy_class in hwy_classes:

    df_class = df[df['HighwayClassCode'] == hwy_class]
    df_class.reset_index(drop=True, inplace=True)
    hwy_class_desc = df_class.HighwayClassDescription.iloc[0]

    plt.scatter(df_class['AADT'],df_class[crash_col])

    # annotating the road names of the highest AADT and highest Crash count segments
    aadt_max = df_class['AADT'].max()  # x value of aadt max
    aadt_max_index = df_class[df_class['AADT'] == aadt_max].index.values[0]
    aadt_max_crash = df_class[crash_col].iloc[aadt_max_index]
    aadt_max_name = df_class['RouteNames'].iloc[aadt_max_index]
    aadt_max_parish = df_class['Parishes'].iloc[aadt_max_index]
    plt.annotate(text=f'{aadt_max_name} \n {aadt_max_parish}',
                 xy=(aadt_max,aadt_max_crash))

    crash_max = df_class[crash_col].max()  # y value of crash max
    crash_max_index = df_class[df_class[crash_col] == crash_max].index.values[0]
    crash_max_aadt = df_class['AADT'].iloc[crash_max_index]
    crash_max_name = df_class['RouteNames'].iloc[crash_max_index]
    crash_max_parish = df_class['Parishes'].iloc[crash_max_index]
    plt.annotate(text=f'{crash_max_name} \n {crash_max_parish}',
                 xy=(crash_max_aadt,crash_max))

    plt.xlabel('AADT')
    plt.ylabel(crash_col)
    plt.title(f'Highway Class: {hwy_class_desc}')
    plt.show()

    time.sleep(0.1)


