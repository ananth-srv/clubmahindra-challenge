import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import KFold

categorical_vars = ['reservationstatusid_code', 'room_type_booked_code', 'member_age_buckets', 'state_code_resort',
                    'state_code_residence', 'season_holidayed_code', 'resort_id']

def format_data(inp_df):

    inp_df["checkin_date"] = pd.to_datetime(inp_df['checkin_date'],format="%d/%m/%y")
    inp_df["checkout_date"] = pd.to_datetime(inp_df['checkout_date'],format="%d/%m/%y")
    inp_df["booking_date"] = pd.to_datetime(inp_df['booking_date'],format="%d/%m/%y")
    inp_df["advance_booking"] = (inp_df['checkin_date'] - inp_df["booking_date"]).dt.days
    inp_df["days_stayed"] = (inp_df['checkout_date'] - inp_df['checkin_date']).dt.days
    inp_df["weekdays_stayed"] = inp_df.apply(lambda row: np.busday_count(row['checkin_date'],row['checkout_date']), axis=1)
    inp_df["weekends_stayed"] = inp_df["days_stayed"] - inp_df["weekdays_stayed"]
    inp_df["dropped_days"] = inp_df["roomnights"] - inp_df["days_stayed"]
    inp_df["checkin_month"] = inp_df['checkin_date'].dt.month
    inp_df["checkin_week"] = inp_df['checkin_date'].dt.week
    inp_df["checkin_year"] = inp_df['checkin_date'].dt.year
    inp_df["checkout_month"] = inp_df['checkout_date'].dt.month
    inp_df["total_pax_days"] = inp_df['days_stayed'] * inp_df['total_pax']

    calc_mean = inp_df.groupby(['resort_id'], axis=0).agg(
        {"total_pax":"mean","days_stayed":"mean","advance_booking":"mean","total_pax_days":"mean"}).reset_index()
    calc_mean.columns = ['resort_id','totalpax_mean',"days_stayed_resmean","advance_booking_resmean","totpaxdays_resmean"]
    inp_df = inp_df.merge(calc_mean,on=['resort_id'],how='left')

    calc_mean = inp_df.groupby(['resort_id','checkin_month'], axis=0).agg(
        {"total_pax":"mean"}).reset_index()
    calc_mean.columns = ['resort_id','checkin_month','totalpax_chkmean']
    inp_df = inp_df.merge(calc_mean,on=['resort_id','checkin_month'],how='left')

    calc_mean = inp_df.groupby(['memberid'], axis=0).agg(
        {"total_pax":"mean","days_stayed":"mean","advance_booking":"mean","reservation_id":"count",
         "roomnights":"mean","numberofadults":"mean","numberofchildren":"mean","weekends_stayed":"mean", "weekdays_stayed":"mean",
         "total_pax_days":"mean"}).reset_index()
    calc_mean.columns = ['memberid','totalpax_memmean',"days_stayed_memmean","advance_booking_memmean",
                         "res_memcnt","roomnights_memmean","adults_memmean","child_memmean","weekends_memmean","weekdays_memmean",
                         "totpaxdays_memmean"]
    inp_df = inp_df.merge(calc_mean,on=['memberid'],how='left')

    calc_mean = inp_df.groupby(['memberid'], axis=0).agg(
        {"days_stayed":"sum","advance_booking":"sum","total_pax":"sum","numberofadults":"sum","numberofchildren":"sum",
         "weekends_stayed":"sum","total_pax_days":"sum"}).reset_index()
    calc_mean.columns = ['memberid',"days_stayed_memsum","advance_booking_memsum","total_pax_memsum","adults_memsum","child_memsum",
                         "weekend_memsum","totpaxdays_memsum"]
    inp_df = inp_df.merge(calc_mean,on=['memberid'],how='left')

    calc_mean = inp_df.groupby(['memberid','resort_id'], axis=0).agg(
        {"total_pax":"mean","days_stayed":"mean","roomnights":"mean","reservation_id":"count","total_pax_days":"mean"}).reset_index()
    calc_mean.columns = ['memberid','resort_id','totalpax_memresmean',"days_stayed_memresmean","roomnights_memresmean",
                         "book_memrescnt","totpaxdays_memresmean"]
    inp_df = inp_df.merge(calc_mean,on=['memberid','resort_id'],how='left')

    calc_mean = inp_df.groupby(['memberid','resort_id','checkin_month'], axis=0).agg(
        {"total_pax":"mean","reservation_id":"count","total_pax_days":"mean"}).reset_index()
    calc_mean.columns = ['memberid','resort_id','checkin_month','totalpax_memreschkmean',
                         "book_memreschkcnt","totpaxdays_memreschkmean"]
    inp_df = inp_df.merge(calc_mean,on=['memberid','resort_id','checkin_month'],how='left')

    inp_df["passengers_dropped"] = inp_df["total_pax"] - (inp_df["numberofadults"] + inp_df["numberofchildren"])
    drop = ['checkin_date','checkout_date','booking_date','memberid']
    inp_df = inp_df.drop(drop, axis=1)
    inp_df = inp_df.fillna(-1)
    column_names = inp_df.columns
    for i in column_names:
        if inp_df[i].dtype == 'object' and i not in ['reservation_id','type']:
            lbl = LabelEncoder()
            lbl.fit(list(inp_df[i].values))
            inp_df[i] = lbl.transform(list(inp_df[i].values))

    for var in categorical_vars:
        inp_df[var] = inp_df[var].astype('category')
    return inp_df

def remove_outliers(inp_df):
    inp_df = inp_df.drop(inp_df[inp_df['roomnights'] < 0].index)
    inp_df = inp_df.drop(inp_df[inp_df['advance_booking'] < 0].index)
    inp_df = inp_df.drop(inp_df[inp_df['days_stayed'] < 0].index)
    inp_df = inp_df.drop(inp_df[(inp_df["numberofadults"] + inp_df["numberofchildren"]) == 0].index)
    inp_df = inp_df.drop(inp_df[inp_df['total_pax'] == 0].index)

    return inp_df


train = pd.read_csv("club/train.csv")
train['type'] = "train"
test = pd.read_csv("club/test.csv")
test["amount_spent_per_room_night_scaled"] = 0
test["type"] = "test"
final_df = pd.concat([train,test],axis=0)
final_df = format_data(final_df)

test = final_df[final_df["type"]=="test"]
test = test.drop(['type'],axis=1)
train = final_df[final_df["type"]=="train"]
train = train.drop(['type'],axis=1)
train = remove_outliers(train)

X = train.drop(['reservation_id','amount_spent_per_room_night_scaled'], axis=1)
feature_set = X.columns
X = X.values
y = train['amount_spent_per_room_night_scaled'].values
nrounds = 4000

params = {'metric': 'rmse', 'learning_rate': 0.02, 'max_depth': 6, 'objective': 'regression',"n_estimators": 10000,
          'feature_fraction': 0.9, 'bagging_fraction': 1, 'lambda_l1': 2, 'lambda_l2': 4, 'num_leaves': 600,
          'min_gain_to_split': .1}

xgb_params = {'eta': 0.02, 'max_depth': 6, 'subsample':0.8, 'colsample_bytree': 0.9,
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}

sub = test[['reservation_id']]
sub['amount_spent_per_room_night_scaled'] = 0
valid = 0
count = 0
kfold = 5

skf = KFold(n_splits=kfold)
skf.get_n_splits(X,y)

for train_index, test_index in skf.split(X,y):
    count = count + 1
    print(' lgb kfold: {}  of  {}'.format(count, 5))
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]

    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=Y_train), nrounds,
                          lgb.Dataset(X_test, label=Y_test), verbose_eval=50, early_stopping_rounds=50)

    sub['amount_spent_per_room_night_scaled'] += lgb_model.predict(test[feature_set].values,
                                                num_iteration=lgb_model.best_iteration)

    d_train = xgb.DMatrix(X_train, Y_train)
    d_valid = xgb.DMatrix(X_test, Y_test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    xgb_model = xgb.train(xgb_params, d_train, nrounds, watchlist, early_stopping_rounds=50,
                          maximize=False, verbose_eval=50)

    sub['amount_spent_per_room_night_scaled'] += xgb_model.predict(xgb.DMatrix(test[feature_set].values),
                                                                   ntree_limit=xgb_model.best_ntree_limit)

    model = CatBoostRegressor(learning_rate=0.02, depth=6, iterations=5000,
                              eval_metric="RMSE", verbose=True,bootstrap_type="Bernoulli")

    fit_model = model.fit(X_train, Y_train, eval_set=(X_test, Y_test), use_best_model=True)

    sub['amount_spent_per_room_night_scaled'] += fit_model.predict(test[feature_set].values)


sub['amount_spent_per_room_night_scaled'] = sub['amount_spent_per_room_night_scaled']/(3 * kfold)
sub.to_csv('club/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)