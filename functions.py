import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OneHotEncoder
import sklearn.preprocessing as pp
import datetime
#from pyzipcode import ZipCodeDatabase
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.cross_validation import cross_val_predict
import statsmodels.api as sm
from sklearn.cross_validation import Bootstrap
from sklearn.ensemble import RandomForestRegressor


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


def make_squared(df, features):
    for feat1 in features:
        df[feat1 + '_sq'] = df[feat1]**2
        #df[feat1 + '_cu'] = df[feat1]**3
        df[feat1 + '_sqrt'] = np.sqrt(df[feat1])
        df['ln_' + feat1] = np.log(df[feat1])
    return df


def scale_skiers(df, range, ref_column):
    df_scaled = df.copy().astype(float)
    minmax_scale = pp.MinMaxScaler(feature_range=range, copy=True)
    minmax_scale_fit = minmax_scale.fit(df[ref_column])
    scaled_skiers = minmax_scale_fit.transform(df[ref_column])
    return scaled_skiers


def pp_standard_scaler(X):
    standard_scale = pp.StandardScaler()
    standard_scale_fit = standard_scale.fit(X)
    return standard_scale_fit.transform(X), standard_scale_fit


def scale_features(df):
    df_scaled = df.copy().astype(float)
    features = list(df.columns.values)
    #Center feature values around zero and make them all have variance on the same order.
    df_scaled_array, standard_scale_fit = pp_standard_scaler(df)
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


def make_midweek_col(df_all):
    midweek = []
    for idx, row in df_all.iterrows():
        if row.day_1 == 1. or row.day_2 == 1. or row.day_3 == 1:
            midweek.append(1)
        else:
            midweek.append(0)
    df_all['midweek'] = midweek
    return df_all


def get_holiday_list(start_yr, start_mon, start_day, end_yr, end_mon, end_day):
    calendar = USFederalHolidayCalendar()
    holiday_list = calendar.holidays(datetime.datetime(start_yr, 11, 1), datetime.datetime(end_yr, 5, 1))
    holidays = []
    for holiday in holiday_list:
        holidays.append(holiday)
    return holidays


def get_holiday_names(holidays, vet_day, thanksgiv, xmas, mlk_day, pres_day):
    vet_day.append(holidays[0])
    thanksgiv.append(holidays[1])
    xmas.append(holidays[2])
    mlk_day.append(holidays[4])
    pres_day.append(holidays[5])
    return vet_day, thanksgiv, xmas, mlk_day, pres_day


def make_thanks_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = (idx.date() - holiday.date()).days
        if diff >= -5 and diff <= 4:
            is_holiday.append(1)
        else:
            is_holiday.append(0)
    df[hol_name] = is_holiday
    return df


def make_pres_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = (idx.date() - holiday.date()).days
        if diff > -7 and diff <= 0:
            is_holiday.append(1)
        else:
            is_holiday.append(0)
    df[hol_name] = is_holiday
    return df


def make_xmas_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = (idx.date() - holiday.date()).days
        if diff < 8 and diff > -2:
            is_holiday.append(1)
        else:
            is_holiday.append(0)
    df[hol_name] = is_holiday
    return df
    df[hol_name] = is_holiday
    return df

def make_before_xmas_column(df, holiday, hol_name):
    is_holiday = []
    for idx, row in df.iterrows():
        diff = (idx.date() - holiday.date()).days
        if diff >= -6 and diff <= -2:
            is_holiday.append(1)
        else:
            is_holiday.append(0)
    df[hol_name] = is_holiday
    return df
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

def call_make_holiday_columns(df_all, vet_day, thanksgiv, xmas, mlk_day, pres_day):
    df_all = make_mlk_column(df_all, vet_day, 'vet_day')
    df_all = make_thanks_column(df_all, thanksgiv, 'thanksgiv')
    df_all = make_xmas_column(df_all, xmas, 'xmas')
    df_all = make_mlk_column(df_all, mlk_day, 'mlk_day')
    df_all = make_pres_column(df_all, pres_day, 'pres_day')
    df_all = make_before_xmas_column(df_all, xmas, 'before_xmas')
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


def make_spring_break_col(df):
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


def forward_selection_step(model, X_tr, y_tr, n_feat, features, best_features, y_tr_mar, y_tr_apr, y_tr_dec, mar_pos, apr_pos, dec_pos):
    #initialize min_MSE with a very large number
    min_score = 1000000000000000000
    next_feature = ''
    for f in features:
        feat = best_features + [f]
        mdl = model.fit(X_tr[feat], y_tr)
        y_pred = cross_val_predict(mdl, X_tr[feat], y_tr, cv=20)
        y_pred_dec = y_pred[dec_pos]
        y_pred_mar = y_pred[mar_pos]
        y_pred_apr = y_pred[apr_pos]
        score_cv_step = mean_squared_error(y_tr, y_pred)  + mean_squared_error(y_tr[y_tr<20], y_pred[y_tr<20]) + mean_squared_error(y_tr[y_tr>60], y_pred[y_tr>60])
        # +  mean_squared_error(y_tr_dec, y_pred_dec)
        MSE_step = mean_squared_error(y_tr, y_pred)
        if score_cv_step < min_score:
            min_score = score_cv_step
            next_feature = f
            score_cv = round(min_score, 1)
            MSE = round(MSE_step, 1)
    return next_feature, MSE


def forward_selection_lodo(model, X_tr, y_tr, n_feat, features, y_tr_mar, y_tr_apr, y_tr_dec, mar_pos, apr_pos, dec_pos):
    #initialize the best_features list with the base features to force their inclusion
    best_features = []
    RMSE = []
    #r2 = []
    while len(features) > 0 and len(best_features) < n_feat:
        next_feature, MSE_feat = forward_selection_step(model, X_tr, y_tr, n_feat, features, best_features, y_tr_mar, y_tr_apr, y_tr_dec, mar_pos, apr_pos, dec_pos)
        #add the next feature to the list
        best_features += [next_feature]
        RMSE_features = round(np.sqrt(MSE_feat), 1)
        RMSE.append(RMSE_features)
        #r2.append(MSE_feat)
        print 'Next best Feature: ', next_feature, ',','RMSE: ', RMSE_features, "#:", len(best_features)
        #print 'Next best Feature: ', next_feature, ',','R2: ', r2, "#:", len(best_features)
        #remove the added feature from the list
        features.remove(next_feature)
    print "Best Features: ", best_features
    return best_features, RMSE


def find_best_lambda(Model, features, X_tr, y_tr, min_lambda, max_lambda, mult_factor):
    lambda_ridge = []
    mean_score_lambda = []
    i = min_lambda
    n = 1
    coefs = []
    while i < max_lambda:
        #print 'lambda:', i
        model = Model(alpha=i)
        lso = model.fit(X_tr, y_tr)
        y_pred = cross_val_predict(lso, X_tr, y_tr, cv=10)
        mean_score_lambda.append(round(np.sqrt(mean_squared_error(y_tr, y_pred)), 0))
        #print 'score:', mean_score_lambda[n-1]
        #record the lambda value for this run
        lambda_ridge.append(i)
        #record the coefficients for this lambda value
        coefs.append(model.coef_)
        i = i * mult_factor
        n += 1

    #find the lambda value (that produces the lowest cross-validation MSE)
    best_lambda = lambda_ridge[mean_score_lambda.index(min(mean_score_lambda))]

    print 'Best Lambda:', round(best_lambda, 0)
    return best_lambda, lambda_ridge, coefs, mean_score_lambda


def get_holdout_RMSE(model, feat, df_tr, df_h, ref_column):
    df_hold = pd.concat([df_h[ref_column], df_h[feat]], axis=1)
    X_tr = df_tr[feat]
    y_tr = df_tr[ref_column].values
    X_h = df_h[feat]
    y_h = df_h[ref_column].values
    mdl = model.fit(X_tr, y_tr)
    cv_pred = cross_val_predict(model, X_tr, y_tr, cv = 10)
    pred_h = mdl.predict(X_h)
    df_cv = df_tr
    df_cv['pred'] = cv_pred
    df_hold['pred'] = pred_h
    RMSE_h = round(np.sqrt(mean_squared_error(y_h, pred_h)), 1)
    RMSE_CV = round(np.sqrt(mean_squared_error(y_tr, cv_pred)), 1)
    print 'CV RMSE:', RMSE_CV, ', ', 'Holdout RMSE:', RMSE_h
    return RMSE_h, RMSE_CV, df_hold, df_cv


def find_training_and_hold_sets(df_tr, df_h, features, ref_column):
    X_tr = df_tr[features]
    y_tr = df_tr[ref_column].values
    X_h = df_h[features]
    y_h = df_h[ref_column].values
    return X_tr, y_tr, X_h, y_h


def add_pred_and_conf_int_to_df(df_pred, df_fut):
    pred_mean = []
    pred_std = []
    for idx, row in df_pred.iterrows():
        pred_mean.append(round(row.mean(),0))
        pred_std.append(round(row.std(),1))
    df_fut['pred'] = pred_mean
    df_fut['st_dev'] = pred_std
    df_fut['lower'] = df_fut['pred'] - df_fut['st_dev']*2
    df_fut['upper'] = df_fut['pred'] + df_fut['st_dev']*2
    return df_fut


def run_bootstrap_model(df_cv, model, best_features, X_fut, ref_column):
    bs = Bootstrap(len(df_cv), n_iter=1000, train_size=int(len(df_cv)*3/4), test_size=int(len(df_cv)*1/4))
    count = 1
    first = True
    for train_index, test_index in bs:
        X_tr = df_cv.ix[train_index][best_features]
        y_tr = df_cv.ix[train_index][ref_column]
        mdl = model.fit(X_tr, y_tr)
        pred = mdl.predict(X_fut)
        if first:
            df_pred = pd.DataFrame(pred)
            first = False
        else:
            df_pred[count] = pred
        count += 1
    return df_pred


if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
