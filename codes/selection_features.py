from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd

from boruta import BorutaPy
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

data = pd.read_csv('../datasets/titanic.csv')

data.drop(['Ticket', 'PassengerId'], axis=1, inplace=True)

#CONVERTER GÊNERO PARA 0s e 1s.
gender_mapper = {'male': 0, 'female': 1}
data['Sex'].replace(gender_mapper, inplace=True)

#CRIAR UMA COLUNA COM O Title
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].strip().split(' ')[0])
data['Title'] = [0 if x in ['Mr.', 'Miss.', 'Mrs.'] else 1 for x in data['Title']]
data = data.rename(columns={'Title': 'Title_Unusual'})
data.drop('Name', axis=1, inplace=True)

data['Cabin_Known'] = [0 if str(x) == 'nan' else 1 for x in data['Cabin']]
data.drop('Cabin', axis=1, inplace=True)

emb_dummies = pd.get_dummies(data['Embarked'], drop_first=True, prefix='Embarked')
data = pd.concat([data, emb_dummies], axis=1)
data.drop('Embarked', axis=1, inplace=True)

input_column = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin_Known', 'Embarked_Q', 'Embarked_S']
target = 'Survived'

imputer = KNNImputer(n_neighbors=3)
X_imputed = imputer.fit_transform(data)

data_imputed = pd.DataFrame(X_imputed,columns=data.columns)
forest = RandomForestClassifier(class_weight='balanced', max_depth=5)

def get_columns_by_boruta():
    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, random_state=42)
    feat_selector.fit(data_imputed[input_column].values, data_imputed[target].values)

    # check ranking of features
    return feat_selector.ranking_
def get_columns_by_rfe():
    rfe = RFE(estimator=forest, n_features_to_select=5)
    rfe.fit(data_imputed[input_column].values, data_imputed[target].values)
    
    return rfe.ranking_
ranking_boruta = get_columns_by_boruta()
ranking_rfe = get_columns_by_rfe()