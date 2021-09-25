from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime

df = pd.read_csv('/home/nobrega/Documentos/archive/ANSUR II MALE Public.csv', encoding='latin').drop(
    ['subjectid', 'weightkg'], axis=1)

df = df.select_dtypes(include='number')

X, y = df.iloc[:, :-1], df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1121218
)

X_train_std = (X_train - X_train.mean() )/X_train.std()
X_test_std = (X_test - X_test.mean())/X_test.std()

start = datetime.now()
regressor = RandomForestRegressor().fit(X_train_std, y_train)
print("Trainign R-squared:", regressor.score(X_train_std, y_train))
print("Testing R-squared:",regressor.score(X_test_std, y_test))
print('Duration: {}'.format(datetime.now() - start))

rfecv = RFECV(
    estimator=LinearRegression(),
    min_features_to_select=5,
    step=6,
    n_jobs=4,
    scoring="r2",
    cv=7,
    verbose=0
)

rfecv.fit(X_train_std, y_train)

start = datetime.now()
regressor_filtered = RandomForestRegressor().fit(X_train_std[X_train.columns[rfecv.support_]], y_train)

print("Trainign R-squared:", regressor_filtered.score(X_train_std[X_train.columns[rfecv.support_]], y_train))
print("Testing R-squared:",regressor_filtered.score(X_test_std[X_test.columns[rfecv.support_]], y_test))
print('Duration: {}'.format(datetime.now() - start))