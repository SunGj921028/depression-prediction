from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(train, test):
    # Define categorical and numerical columns
    #! Below have some part is forked from GPUtrain.py written by Buffeet
    # categorical_cols = [
    #     'Gender', 'City', 'Working Professional or Student',
    #     'Sleep Duration', 'Dietary Habits', 'Family History of Mental Illness'
    # ]
    sleep_mapping = {
        "More than 8 hours":9,
        'Less than 5 hours':4,
        '5-6 hours':5.5,
        '7-8 hours':7.5,
        '1-2 hours':1.5,
        '6-8 hours':7,
        '4-6 hours':5,
        '6-7 hours':6.5,
        '10-11 hours':10.5,
        '8-9 hours':8.5,
        '9-11 hours':10,
        '2-3 hours':2.5,
        '3-4 hours':3.5,
        'Moderate':6,
        '4-5 hours':4.5,
        '9-6 hours':7.5,
        '1-3 hours':2,
        '1-6 hours':4,
        '8 hours':8,
        '10-6 hours':8,
        'Unhealthy': 3,
        'Work_Study_Hours':6,
        '3-6 hours':3.5,
        '9-5':7,
        '9-5 hours': 7,
        '49 hours': 7,
        '35-36 hours': 5,
        '40-45 hours': 6,
        'Pune': 6,
        'Indore': 6,
        '45': 6,
        '55-66 hours': 8.5,
        'than 5 hours': 5,
        '45-48 hours': 6.5,
        'No': 6,
        'Sleep_Duration': 6,
    }
    # Apply mapping to both train and test data
    train['Sleep Duration'] = train['Sleep Duration'].map(sleep_mapping)
    test['Sleep Duration'] = test['Sleep Duration'].map(sleep_mapping)
    
    dietary_mapping = {
        'Unhealthy': 0,
        'Less Healthy': 0,
        'No Healthy': 0,
        'Less than Healthy': 0,
        'Moderate': 1,
        'Healthy': 2,
        'More Healthy': 2,
        'Yes': 2, 
        'No': 0,
    }
    train['Dietary Habits'] = train['Dietary Habits'].map(dietary_mapping)
    test['Dietary Habits'] = test['Dietary Habits'].map(dietary_mapping)
    
    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
    # numerical_cols = ['Age', 'Work Pressure', 'Financial Stress']
    numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    columns_to_drop_num = ['id', 'Depression']
    columns_to_drop_cat = ['Name', 'City', 'Profession', 'Degree']
    numerical_cols = [col for col in numerical_cols if col not in columns_to_drop_num]
    categorical_cols = [col for col in categorical_cols if col not in columns_to_drop_cat]

    # Data preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    X = train[categorical_cols + numerical_cols]
    X_processed = preprocessor.fit_transform(X)
    
    # 在轉換測試資料之前，檢查每個類別欄位
    print("\n檢查未知類別：")
    for col in categorical_cols:
        train_categories = set(train[col].unique())
        test_categories = set(test[col].unique())
        unknown_categories = test_categories - train_categories
        
        if len(unknown_categories) > 0:
            print(f"\n欄位 '{col}':")
            print("未知類別:", unknown_categories)
    
    test_X = test[categorical_cols + numerical_cols]
    test_X_processed = preprocessor.transform(test_X)

    return X_processed, test_X_processed

# 使用範例
# train_processed, test_processed = preprocess_data(train_df, test_df)
