import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def preprocess_data(file_path, target_column='gold high'):
    # 1. Import Dataset
    df = pd.read_csv(file_path)
    
    # 2. Convert Date Column
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df.drop(columns=['date'], inplace=True)
    
    # 3. Drop High-Null Columns
    df = df.drop(labels=['us_rates_%', 'CPI', 'GDP'], axis=1, errors='ignore')
    
    # 4. Drop Empty Rows (Market Holidays)
    date_cols = ['year', 'month', 'day', 'day_of_week']
    feature_cols = df.columns.difference(date_cols)
    df = df[~df[feature_cols].isna().all(axis=1)]
    
    # 5. Fill Remaining Nulls
    df = df.fillna(df.mean())
    
    # 6. Drop Data Leakage
    leakage = ['gold open', 'gold close', 'gold low']
    df = df.drop(columns=[col for col in leakage if col in df.columns], errors='ignore')
    
    # 7. Drop Redundant Features (Multicollinearity)
    redundant = [
        'oil open', 'oil close', 'oil low', 
        'platinum open', 'platinum close', 'platinum low',
        'nasdaq open', 'nasdaq close', 'nasdaq low',
        'sp500 open', 'sp500 close', 'sp500 low',
        'silver open', 'silver close', 'silver low',
        'sp500 high', 'nasdaq high', 'year', 'platinum high', 'nasdaq high-low'
    ]
    df = df.drop(columns=[col for col in redundant if col in df.columns], errors='ignore')
    
    # 8. Feature Selection by Importance
    X_temp = df.drop(target_column, axis=1)
    y_temp = df[target_column]
    
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_temp, y_temp)
    
    importances = pd.Series(rf.feature_importances_, index=X_temp.columns)
    selected_features = importances[importances > 0.01].index.tolist()
    
    # 9. Final Data Preparation
    X = df[selected_features]
    y = df[target_column]
    
    # 10. Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, selected_features

# Example how to use
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = preprocess_data(r"D:\Project\SML_Submision\Eksperimen_SML_Alamsyah\dataset\financial_regression_raw.csv")
    print("Otomatisasi Preprocessing Finish!")
    print(f"Feature Used: {features}")
    print(f"Data Train Scaled Sample:\n{X_train[:1]}")