import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump

def preprocess(data, target_column, skewness_features, save_path, train_csv_path, test_csv_path):
    # Splitting Data
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split data into group data type
    numerical_features = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    other_numeric = [col for col in numerical_features if col not in skewness_features]
    
    ordinal_features = [
        'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 
        'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level', 'Distance_from_Home'
    ]
    categorical_features = [
        'Extracurricular_Activities', 'Internet_Access', 'School_Type', 
        'Learning_Disabilities', 'Gender'
    ]

    # Create Pipeline
    category_orders = [
        ['Low', 'Medium', 'High'], ['Low', 'Medium', 'High'], ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High'], ['Low', 'Medium', 'High'], ['Negative', 'Neutral', 'Positive'],
        ['High School', 'College', 'Postgraduate'], ['Near', 'Moderate', 'Far']
    ]
    
    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=category_orders, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    nominal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    skewed_numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('transformer', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])

    standard_numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Merge Pipeline Process
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_pipeline, ordinal_features),
            ('nominal', nominal_pipeline, categorical_features),
            ('skewed_num', skewed_numeric_pipeline, skewness_features),
            ('standard_num', standard_numeric_pipeline, other_numeric)
        ],
        remainder='passthrough'
    )

    # Fitting and transform
    print("\nMemulai proses preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print("Preprocessing selesai!")

    # Get Names Column
    processed_column_names = preprocessor.get_feature_names_out()
    
    #Dataframe train data
    df_train_processed = pd.DataFrame(X_train_processed, columns=processed_column_names, index=X_train.index)
    df_train_processed[target_column] = y_train

    #Dataframe test data
    df_test_processed = pd.DataFrame(X_test_processed, columns=processed_column_names, index=X_test)
    df_test_processed[target_column] = y_test

    #Save to csv
    df_train_processed.to_csv(train_csv_path, index=False)
    df_test_processed.to_csv(test_csv_path, index=False)
    print(f"Path Data Train: {train_csv_path}")
    print(f"Path Data Test: {test_csv_path}")

    # Simpan Preprocessor
    dump(preprocessor, save_path)
    print(f"Pipeline preprocessor disimpan di {save_path}")

    # Pastikan fungsi ini mengembalikan 4 nilai
    return X_train_processed, X_test_processed, y_train, y_test