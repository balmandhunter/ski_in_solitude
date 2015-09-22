import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OneHotEncoder
import sklearn.preprocessing as pp
import datetime
from pyzipcode import ZipCodeDatabase
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_predict


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


def pp_standard_scaler(X):
    standard_scale = pp.StandardScaler()
    standard_scale_fit = standard_scale.fit(X)
    return standard_scale_fit.transform(X), standard_scale_fit


def scale_features(df):
    df_prescaled = df.copy().astype(float)
    features = list(df.columns.values)
    #prescale the features
    prescaled, minmax_scale_fit = sci_minmax(df_scaled[features])
    #Center feature values around zero and make them all have variance on the same order.
    df_scaled_array, standard_scale_fit = pp_standard_scaler(df_prescaled)
    df_scaled = pd.DataFrame(df_scaled_array, columns = features)
    index_col = df.index
    df_scaled.set_index(index_col, inplace=True)
    return df_scaled, standard_scale_fit


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
    holiday_list = calendar.holidays(datetime.datetime(2011, 11, 1), datetime.datetime(2015, 5, 1))
    holidays = []
    for holiday in holiday_list:
        holidays.append(holiday)
    return holidays


def get_holiday_names(holidays):
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
        if row.filter(regex="day_4").values == 1 and diff < 0 and diff > -8:
            is_holiday.append(1.6)
        elif row.filter(regex="day_5").values == 1 and diff <= 1 and diff > -7:
            is_holiday.append(1.5)
        elif row.filter(regex="day_6").values ==  1 and diff < 2 and diff > -6:
            is_holiday.append(0.4)
        elif row.filter(regex="day_0").values ==  1 and diff < 3 and diff > -5:
            is_holiday.append(0.6)
        elif row.filter(regex="day_1").values ==  1 and diff < 4 and diff > -4:
            is_holiday.append(0.8)
        elif row.filter(regex="day_2").values ==  1 and diff < 5 and diff > -3:
            is_holiday.append(1)
        elif row.filter(regex="day_3").values ==  1 and diff < 6 and diff > -2:
            is_holiday.append(1.4)
        elif row.filter(regex="day_4").values ==  1 and diff < 7 and diff > -1:
            is_holiday.append(1.7)
        elif row.filter(regex="day_5").values ==  1 and diff < 8 and diff > 0:
            is_holiday.append(0.9)
        elif row.filter(regex="day_6").values ==  1 and diff < 9 and diff > 1:
            is_holiday.append(0)
        else:
            is_holiday.append(0)
    df[hol_name] = is_holiday
    return df


def make_mlk_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = (idx.date() - holiday.date()).days
        if diff < 0 and diff >= -3:
            is_holiday.append(1)
        elif diff == 0:
            is_holiday.append(1)
        else:
            is_holiday.append(0)
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
    df_all = make_mlk_column(df_all, vet_day, 'vet_day')
    df_all = make_thanks_column(df_all, thanksgiv, 'thanksgiv')
    df_all = make_xmas_column(df_all, xmas, 'xmas')
    df_all = make_new_years_column(df_all, new_years, 'new_years')
    df_all = make_mlk_column(df_all, mlk_day, 'mlk_day')
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


def find_season_total_snowfall(df):
    count = 0
    tot_snow = []
    for idx, row in df.iterrows():
        try:
            tot_snow.append(round(row.new_24 + tot_snow[count-1],1))
        except:
            tot_snow.append(round(row.new_24,1))
        count += 1
    df['tot_snow'] = tot_snow
    return df


def day_of_week_col(df):
    day_of_week = []
    for idx, day in df.iterrows():
        day_of_week.append(idx.weekday())
    df['day_of_week'] = day_of_week
    return df


def forward_selection_step(model, X_tr, y_tr, n_feat, features, best_features):
    #initialize min_MSE with a very large number
    min_score = 100000000000000
    next_feature = ''
    for f in features:
        feat = best_features + [f]
        mdl = model.fit(X_tr[feat], y_tr)
        y_pred = mdl.predict(X_tr[feat])
        score_cv_step = mean_squared_error(y_tr, y_pred)
        if score_cv_step < min_score:
            min_score = score_cv_step
            next_feature = f
            score_cv = round(min_score, 1)
    return next_feature, score_cv


def forward_selection_lodo(model, X_tr, y_tr, n_feat, features):
    #initialize the best_features list with the base features to force their inclusion
    best_features = []
    RMSE = []
    while len(features) > 0 and len(best_features) < n_feat:
        next_feature, MSE_feat = forward_selection_step(model, X_tr, y_tr, n_feat, features, best_features)
        #add the next feature to the list
        best_features += [next_feature]
        RMSE_features = round(np.sqrt(MSE_feat), 1)
        RMSE.append(RMSE_features)
        print 'Next best Feature: ', next_feature, ',','RMSE: ', RMSE_features, "#:", len(best_features)
        #remove the added feature from the list
        features.remove(next_feature)
    print "Best Features: ", best_features
    return best_features, RMSE





if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
