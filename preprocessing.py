import pandas
from scipy.sparse.construct import random
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy
# Load dataset
url = "adult.csv"
df = pandas.read_csv(url)

# filling missing values

col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", numpy.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation','relationship', 'gender', 'native-country', 'income'] 
labelEncoder = preprocessing.LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

#droping redundant columns
df=df.drop(['fnlwgt','educational-num'], axis=1)


X = df.values[:, 0:12]
Y = df.values[:,12]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

hyperParams = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=hyperParams, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_
CV_rfc.best_estimator_
CV_rfc.best_score_

rfc = RandomForestClassifier(random_state=42, criterion="gini", max_depth=8, n_estimators=500)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print ("Classification using Random Forest Classifier (best Parameters found using gridSearch\nAccuracy is ", accuracy_score(y_test,y_pred)*100 )

#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(CV_rfc, open("model.pkl","wb"))

