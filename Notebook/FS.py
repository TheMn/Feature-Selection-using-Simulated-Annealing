import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

def map_categorical_variable(label, data_frame):
    le = LabelEncoder()
    mapped_label = le.fit_transform(data_frame[label])
    data_frame[label+'_label'] = mapped_label
    
def select_features(solution, feature_names):
    tmp = [feature_names[i] for i in range(len(solution)) if solution[i]==1]
    feature_names = tmp
    return feature_names
    
def evaluate(solution):
    
    data_frame = pd.read_csv('heart.csv', encoding='utf-8')
    
    categorical_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for cv in categorical_variables:
        map_categorical_variable(cv, data_frame)
    
    data_frame.drop(categorical_variables, axis=1, inplace=True)
    
    feature_names = data_frame.columns.values.tolist()
    feature_names.remove('HeartDisease')
    feature_names = select_features(solution, feature_names)

    X = data_frame[feature_names]
    y = data_frame['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC()
    svm.fit(X_train, y_train)

    return 1-svm.score(X_test, y_test)