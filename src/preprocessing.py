import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml

def load_config(config_path = 'config.yaml'):
    with open(config_path,'r') as f:
        return yaml.safe_load(f)

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df, scale_method='standard'):
    df = df.copy()

    # dropping customerID as not useful for clustering
    if 'CustomerID' in df.columns:
        df.drop(columns = ['CustomerID'], inplace=True)

    # converting gender to numeric
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    #selecting numeric columns for scaling
    numeric_cols = df.select_dtypes(include=['float64','int64','int32']).columns.tolist()

    
    if scale_method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess_pipeline(config):
    data_path = config['data']['path']
    scale_method = config['preprocessing']['scale_method']

    df = load_data(data_path)
    df_preprocessed = preprocess_data(df, scale_method)

    return df, df_preprocessed

if __name__ == "__main__":
    config = load_config()
    raw_df, processed_df = preprocess_pipeline(config)
    print("raw",raw_df.shape)
    print("processed",processed_df.shape)
    print(processed_df.head())