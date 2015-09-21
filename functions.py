import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OneHotEncoder
import sklearn.preprocessing as pp
import datetime
from pyzipcode import ZipCodeDatabase
from pandas.tseries.holiday import USFederalHolidayCalendar


def choose_date_range(start_yr, start_mon, start_day, end_yr, end_mon, end_day, df_traf):
    return df_traf[(df_traf.index >= datetime.datetime(start_yr, start_mon, start_day)) & (df_traf.index < datetime.datetime(end_yr, end_mon, end_day))]


def separate_traffic_directions(df_traf):
    return df_traf[df_traf.COUNTDIR == 'P'], df_traf[df_traf.COUNTDIR == 'S']


def sum_morning_evening_and_drop_hourly(df_traf_prim, df_traf_sec):
    df_traf_prim['morning_east'] = df_traf_prim.HOUR0 + df_traf_prim.HOUR1 + df_traf_prim.HOUR3 + df_traf_prim.HOUR4 + df_traf_prim.HOUR5 + df_traf_prim.HOUR6 + df_traf_prim.HOUR7 + df_traf_prim.HOUR8 + df_traf_prim.HOUR9 + df_traf_prim.HOUR10 + df_traf_prim.HOUR11
    df_traf_prim['evening_east'] = df_traf_prim.HOUR12 + df_traf_prim.HOUR13 + df_traf_prim.HOUR14 + df_traf_prim.HOUR15 + df_traf_prim.HOUR16 + df_traf_prim.HOUR17 + df_traf_prim.HOUR18 + df_traf_prim.HOUR19 + df_traf_prim.HOUR20 + df_traf_prim.HOUR21 + df_traf_prim.HOUR22 + df_traf_prim.HOUR23
    df_traf_prim.drop(['HOUR0', 'HOUR1','HOUR2','HOUR3','HOUR4','HOUR5','HOUR6','HOUR7','HOUR8','HOUR9','HOUR10','HOUR11','HOUR12','HOUR13','HOUR14','HOUR15','HOUR16','HOUR17','HOUR18','HOUR19','HOUR20','HOUR21','HOUR22','HOUR23'], inplace = True, axis = 1)
    df_traf_sec['morning_west'] = df_traf_sec.HOUR0 + df_traf_sec.HOUR1 + df_traf_sec.HOUR3 + df_traf_sec.HOUR4 + df_traf_sec.HOUR5 + df_traf_sec.HOUR6 + df_traf_sec.HOUR7 + df_traf_sec.HOUR8 + df_traf_sec.HOUR9 + df_traf_sec.HOUR10 + df_traf_sec.HOUR11
    df_traf_sec['evening_west'] = df_traf_sec.HOUR12 + df_traf_sec.HOUR13 + df_traf_sec.HOUR14 + df_traf_sec.HOUR15 + df_traf_sec.HOUR16 + df_traf_sec.HOUR17 + df_traf_sec.HOUR18 + df_traf_sec.HOUR19 + df_traf_sec.HOUR20 + df_traf_sec.HOUR21 + df_traf_sec.HOUR22 + df_traf_sec.HOUR23
    df_traf_sec.drop(['HOUR0', 'HOUR1','HOUR2','HOUR3','HOUR4','HOUR5','HOUR6','HOUR7','HOUR8','HOUR9','HOUR10','HOUR11','HOUR12','HOUR13','HOUR14','HOUR15','HOUR16','HOUR17','HOUR18','HOUR19','HOUR20','HOUR21','HOUR22','HOUR23'], inplace = True, axis = 1)
    return df_traf_prim, df_traf_sec


def sci_minmax(df_col):
    minmax_scale = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    minmax_scale_fit = minmax_scale.fit(df_col)
    return minmax_scale_fit.transform(df_col), minmax_scale_fit


def convert_zip(zipcode):
    zcdb = ZipCodeDatabase()
    try:
        zipcode = zcdb[zipcode]
        return zipcode.state
    except:
        return np.nan


def create_day_of_week_col(df):
    day_of_week = []
    for idx, day in df.iterrows():
        day_of_week.append(idx.weekday())

    df['day_of_week'] = day_of_week
    return df


def get_holiday_list(start_yr, start_mon, start_day, end_yr, end_mon, end_day):
    calendar = USFederalHolidayCalendar()
    holiday_list = calendar.holidays(datetime.datetime(2014, 11, 1), datetime.datetime(2015, 5, 1))
    holidays = []
    for holiday in holiday_list:
        holidays.append(holiday)
    return holidays


def get_specific_holidays(holidays):
    vet_day = holidays[0]
    thanksgiv = holidays[1]
    xmas = holidays[2]
    new_years = holidays[3]
    mlk_day = holidays[4]
    pres_day = holidays[5]
    return vet_day, thanksgiv, xmas, new_years, mlk_day, pres_day


def make_thanks_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = (idx.date() - holiday.date()).days
        if diff > 7:
            is_holiday.append(0)
        elif diff == 0:
            is_holiday.append(775)
        elif diff == -1:
            is_holiday.append(1622)
        elif diff == -2:
            is_holiday.append(367)
        elif diff == -3:
            is_holiday.append(-33)
        elif diff == -4:
            is_holiday.append(-1098)
        elif diff == -5:
            is_holiday.append(739)
        elif diff == 1:
            is_holiday.append(1094)
        elif diff == 2:
            is_holiday.append(-1576)
        elif diff == 3:
            is_holiday.append(-3029)
        elif diff == 4:
            is_holiday.append(-397)
        else:
            is_holiday.append(0)
    df[hol_name] = is_holiday
    return df


def make_pres_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = (idx.date() - holiday.date()).days
        if diff > 7:
            is_holiday.append(0)
        elif diff == 0:
            is_holiday.append(-3989)
        elif diff == -1:
            is_holiday.append(-2643)
        elif diff == -2:
            is_holiday.append(1519)
        elif diff == -3:
            is_holiday.append(4511)
        elif diff == -4:
            is_holiday.append(1202)
        elif diff == -5:
            is_holiday.append(197)
        elif diff == -6:
            is_holiday.append(-60)
        else:
            is_holiday.append(0)
    df[hol_name] = is_holiday
    return df


def make_xmas_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = (idx.date() - holiday.date()).days
        if row['day_4.0'] ==  1 and diff < 0 and diff > -8:
            is_holiday.append(1.6)
        elif row['day_5.0'] ==  1 and diff <= 1 and diff > -7:
            is_holiday.append(1.5)
        elif row['day_6.0'] ==  1 and diff < 2 and diff > -6:
            is_holiday.append(0.4)
        elif row['day_0.0'] ==  1 and diff < 3 and diff > -5:
            is_holiday.append(0.6)
        elif row['day_1.0'] ==  1 and diff < 4 and diff > -4:
            is_holiday.append(0.8)
        elif row['day_2.0'] ==  1 and diff < 5 and diff > -3:
            is_holiday.append(1)
        elif row['day_3.0'] ==  1 and diff < 6 and diff > -2:
            is_holiday.append(1.4)
        elif row['day_4.0'] ==  1 and diff < 7 and diff > -1:
            is_holiday.append(1.7)
        elif row['day_5.0'] ==  1 and diff < 8 and diff > 0:
            is_holiday.append(0.9)
        elif row['day_6.0'] ==  1 and diff < 9 and diff > 1:
            is_holiday.append(0)
        else:
            is_holiday.append(0)
    df[hol_name] = is_holiday
    return df


def make_holiday_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = abs(idx.date() - holiday.date()).days
        if diff > 7:
            is_holiday.append(0)
        elif diff == 0:
            is_holiday.append(7)
        else:
            is_holiday.append(7-diff)
    df[hol_name] = is_holiday
    return df


def make_new_years_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = abs(idx.date() - holiday.date()).days
        if diff > 6:
            is_holiday.append(0)
        else:
            is_holiday.append(1)
    df[hol_name] = is_holiday
    return df


def call_make_holiday_columns(df_all, vet_day, thanksgiv, xmas, new_years, mlk_day, pres_day):
    df_all = make_holiday_column(df_all, vet_day, 'vet_day')
    df_all = make_thanks_column(df_all, thanksgiv, 'thanksgiv')
    df_all = make_xmas_column(df_all, xmas, 'xmas')
    df_all = make_new_years_column(df_all, new_years, 'new_years')
    df_all = make_holiday_column(df_all, mlk_day, 'mlk_day')
    df_all = make_pres_column(df_all, pres_day, 'pres_day')
    return df_all


def make_spring_trailing_weeks(df, holiday, hol_name):
    #3/8-3/30 is spring break
    is_holiday = []
    for idx, row in df.iterrows():
        if idx.date() >= datetime.datetime(2015,3,1,0,0,0).date() and idx.date() <= datetime.datetime(2015,3,1,0,0,0).date():
            is_holiday.append(1)
        elif idx.date() >= datetime.datetime(2015,3,31,0,0,0).date() and idx.date() <= datetime.datetime(2015,4,5,0,0,0).date():
            is_holiday.append(1)
        else:
            is_holiday.append(0)
    df['spring_break_ends'] = is_holiday
    return df


def make_spring_break_col(df, holiday, hol_name):
    #3/8-3/30 is spring break
    is_holiday = []
    for idx, row in df.iterrows():
        if idx.date() >= datetime.datetime(2015,3,8,0,0,0).date() and idx.date() <= datetime.datetime(2015,3,30,0,0,0).date():
            is_holiday.append(1)
        else:
            is_holiday.append(0)
    df['spring_break'] = is_holiday
    return df


def find_lag_integral(df, time_range, data_col):
    area_curve = []
    for i in range(0,time_range):
        area_curve.append(np.nan)
    for i in range(time_range,len(df[data_col])):
        top = i - time_range
        area_curve.append(df.new_24[top:i].sum())
    return area_curve


def find_season_total_snowfall(df_for):
    count = 0
    tot_snow = []
    for idx, row in df_for.iterrows():
        try:
            if (idx - df_for.index[count -1]).days < 30:
                tot_snow.append(row.new_24 + tot_snow[count-1])
        except:
            tot_snow.append(0)
        count += 1
    df_for['tot_snow'] = tot_snow
    return df_for





if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
