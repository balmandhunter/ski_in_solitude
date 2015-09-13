import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder as OneHotEncoder
import sklearn.preprocessing as pp

import datetime

def choose_date_range(start_yr, start_mon, start_day, end_yr, end_mon, end_day, df_traf):
    return df_traf[(df_traf.index >= datetime.datetime(start_yr, start_mon, start_day)) & (df_traf.index < datetime.datetime(end_yr, end_mon, end_day))]

def separate_traffic_directions(df_traf):
    return df_traf[df_traf.COUNTDIR == 'P'], df_traf[df_traf.COUNTDIR == 'S']

def sum_morning_evening_and_drop_hourly(df_traf_prim, df_traf_sec):
    df_traf_prim['morning_away'] = df_traf_prim.HOUR0 + df_traf_prim.HOUR1 + df_traf_prim.HOUR3 + df_traf_prim.HOUR4 + df_traf_prim.HOUR5 + df_traf_prim.HOUR6 + df_traf_prim.HOUR7 + df_traf_prim.HOUR8 + df_traf_prim.HOUR9 + df_traf_prim.HOUR10 + df_traf_prim.HOUR11 
    df_traf_prim['evening_away'] = df_traf_prim.HOUR12 + df_traf_prim.HOUR13 + df_traf_prim.HOUR14 + df_traf_prim.HOUR15 + df_traf_prim.HOUR16 + df_traf_prim.HOUR17 + df_traf_prim.HOUR18 + df_traf_prim.HOUR19 + df_traf_prim.HOUR20 + df_traf_prim.HOUR21 + df_traf_prim.HOUR22 + df_traf_prim.HOUR23
    df_traf_prim.drop(['HOUR0', 'HOUR1','HOUR2','HOUR3','HOUR4','HOUR5','HOUR6','HOUR7','HOUR8','HOUR9','HOUR10','HOUR11','HOUR12','HOUR13','HOUR14','HOUR15','HOUR16','HOUR17','HOUR18','HOUR19','HOUR20','HOUR21','HOUR22','HOUR23'], inplace = True, axis = 1)
    df_traf_sec['morning_to'] = df_traf_sec.HOUR0 + df_traf_sec.HOUR1 + df_traf_sec.HOUR3 + df_traf_sec.HOUR4 + df_traf_sec.HOUR5 + df_traf_sec.HOUR6 + df_traf_sec.HOUR7 + df_traf_sec.HOUR8 + df_traf_sec.HOUR9 + df_traf_sec.HOUR10 + df_traf_sec.HOUR11 
    df_traf_sec['evening_to'] = df_traf_sec.HOUR12 + df_traf_sec.HOUR13 + df_traf_sec.HOUR14 + df_traf_sec.HOUR15 + df_traf_sec.HOUR16 + df_traf_sec.HOUR17 + df_traf_sec.HOUR18 + df_traf_sec.HOUR19 + df_traf_sec.HOUR20 + df_traf_sec.HOUR21 + df_traf_sec.HOUR22 + df_traf_sec.HOUR23
    df_traf_sec.drop(['HOUR0', 'HOUR1','HOUR2','HOUR3','HOUR4','HOUR5','HOUR6','HOUR7','HOUR8','HOUR9','HOUR10','HOUR11','HOUR12','HOUR13','HOUR14','HOUR15','HOUR16','HOUR17','HOUR18','HOUR19','HOUR20','HOUR21','HOUR22','HOUR23'], inplace = True, axis = 1)
    return df_traf_prim, df_traf_sec

def sci_minmax(df_col):
    minmax_scale = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    minmax_scale_fit = minmax_scale.fit(df_col)
    return minmax_scale_fit.transform(df_col), minmax_scale_fit





if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))