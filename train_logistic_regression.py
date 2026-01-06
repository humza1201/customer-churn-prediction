import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

def my_pipline(num_attr, cat_attr):
    num_pipline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("stan_scalar",StandardScaler())
        ]
    )
    
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder" , OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    
    full_pipeline = ColumnTransformer(
        [
            ("num" , num_pipline, num_attr),
            ("cat", cat_pipeline, cat_attr)
        ]
    )
    
    return full_pipeline

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    data = pd.read_csv('telco-customer-churn.csv')
    df = pd.DataFrame(data)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0

    ten_bins = [0,12, 24, 48, 72]
    df['tenure_bin'] = pd.cut(df['tenure'], bins=ten_bins, labels=["0–12", "12–24", "24–48", "48–72"], right=False)

    bins = [0, 35, 70, 120]
    df['monthly_bins'] = pd.cut(df['MonthlyCharges'] , bins = bins, labels=["0–35", "35–70", "70–120"], right=False)

    split = StratifiedShuffleSplit(n_splits = 1, test_size=0.3 , random_state=42)
    for train_ind , test_ind in split.split(df , df['Churn']):
        strata_train_set = df.iloc[train_ind]
        strata_test_set = df.iloc[test_ind]

    train_set = strata_train_set.copy()
    test_set = strata_test_set.copy()

    train_set = train_set.drop([
        'tenure_bin',
        'monthly_bins'
    ] ,axis=1)

    test_set = test_set.drop([
        'tenure_bin',
        'monthly_bins'
    ] ,axis=1)

    train_set_features = train_set.drop(columns=['Churn'])
    train_set_labels = train_set['Churn']

    test_set_features = test_set.drop(columns=['Churn'])
    test_set_labels = test_set['Churn']
        
    train_set_features['SeniorCitizen'] = train_set_features['SeniorCitizen'].astype('object')
    test_set_features['SeniorCitizen'] = test_set_features['SeniorCitizen'].astype('object')

    num_attr = train_set_features.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_attr = [col for col in train_set_features.columns 
            if col not in num_attr + ['customerID']]

    test_num_attr = test_set_features.select_dtypes(include=['int64','float64']).columns.tolist()
    test_cat_attr = cat_attr

    full_pipeline = my_pipline(num_attr, cat_attr)
    complete_preprocessed_data = full_pipeline.fit_transform(train_set_features)
    complete_preprocessed_test_data = full_pipeline.transform(test_set_features)

    lrg = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000,
        random_state=42)
    lrg.fit(complete_preprocessed_data, train_set_labels)
    
    joblib.dump(lrg , MODEL_FILE)
    joblib.dump(full_pipeline , PIPELINE_FILE)

    print("model and pipeline have been trained and saved!")


else:
    lrg = joblib.load(MODEL_FILE)
    full_pipeline = joblib.load(PIPELINE_FILE)
    
    test_data = pd.read_csv("telco-customer-churn-test.csv").drop("Churn",axis = 1, errors = 'ignore')
    test_data['TotalCharges'] = pd.to_numeric(test_data['TotalCharges'], errors='coerce')
    test_data.loc[test_data['tenure'] == 0, 'TotalCharges'] = 0
    test_data = test_data.drop(columns=['customerID'], errors='ignore')
    
    complete_preprocessed_test_data = full_pipeline.transform(test_data)  
    lrg_pred = lrg.predict(complete_preprocessed_test_data)
    test_data['my_predictions'] = lrg_pred
    test_data.to_csv("output.csv", index= False)
    