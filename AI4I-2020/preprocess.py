## This file is created once during manual setup 
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def unknown_fail_check(row): return ((row['Machine failure'] == 1)
                                     & (row['RNF'] == 0)
                                     & (row['HDF'] == 0)
                                     & (row['TWF'] == 0)
                                     & (row['PWF'] == 0)
                                     & (row['OSF'] == 0))

def pass_yet_fail_check(row): return (row['Machine failure'] == 0) & ((row['RNF'] == 1)
                                                                     | (row['HDF'] == 1)
                                                                     | (row['TWF'] == 1)
                                                                     | (row['PWF'] == 1)
                                                                     | (row['OSF'] == 1))
def preprocessing(df):
    print("# Preprocessing")
    df['Type'] = df['Type'].astype('category')
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df['Type'] = df['Type'].map(type_mapping).astype('int')
    print(" Type  Unique Values after encoding: ", df['Type'].unique())
    df.drop(columns=['UDI', 'Product ID'], inplace=True)
    print(f"shape of data after dropping columns {df.shape}")
    df.columns = [col.replace("[","(").replace("]",")") for col in df.columns.values]
    print("DF columns after clean up", df.columns)
    print("## Handle Duplicates") 
    # our original dataset does not have duplicates
    # However, there is no guarantee that production/new data is free of duplicates
    duplicated_row_count = df.duplicated().sum()
    total_row_count = df.shape[0]
    duplicated_row_percentage = (duplicated_row_count/total_row_count*100)
    print(f"Total rows count: {total_row_count}")
    print(f"Duplicated rows count: {duplicated_row_count}")
    print(f"Duplicated rows percentage: {duplicated_row_percentage}")
    df.drop_duplicates(inplace=True)
    print("After removing duplicates rows count:", df.shape[0])
    print("## Handle NULL") 
    print("number of null values : ", df.isnull().sum().sum())
    df.dropna(inplace=True)
    print("After removing null rows count:", df.shape[0])

    passed_although_failed = df[pass_yet_fail_check(df)]
    print(
        f"Number of samples that passed although failed: {len(passed_although_failed)}")
    passed_although_failed.loc[:, ['Machine failure',
                                'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].head(10)
    df['Machine failure'] = np.where(
        pass_yet_fail_check(df), 1, df['Machine failure'])
    passed_although_failed = df[pass_yet_fail_check(df)]
    print(
        f"Number of samples that passed although failed after fix: {len(passed_although_failed)}")

    print(f"Number of machine failures: {df['Machine failure'].sum()}")
    unknown_failures = df[unknown_fail_check(df)]
    print(
        f"Number of failures due to unknown reasons: {len(unknown_failures)}")
    unknown_failures.loc[:, ['Machine failure',
                            'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].head(10)
    df['Machine failure'] = np.where(
        unknown_fail_check(df), 0, df['Machine failure'])
    unknown_failures = df[unknown_fail_check(df)]
    print(
        f"Number of failures due to unknown reasons after fix: {len(unknown_failures)}")
    print("## Add Features") 
    df['Strain (minNm)'] = df['Tool wear (min)'] * df['Torque (Nm)'] 
    df['Power (W)'] = df['Rotational speed (rpm)'] * df['Torque (Nm)'] * 2 * np.pi / 60
    df['Temperature Difference (K)'] = df['Process temperature (K)'] - df['Air temperature (K)']
    labels = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    print("# Splitting into train/test...")
    X = df.drop(columns=labels)
    y = df[labels]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42,  stratify=y['Machine failure']) 
    train=pd.concat([X_train, y_train], axis=1)
    test=pd.concat([X_test, y_test], axis=1)
    return train, test

if __name__ == "__main__":
    # The pipeline will pass arguments to this script.
    # The argument will be used to pass the S3 path of our data.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="path containing data.csv")
    parser.add_argument("--output-train-path", type=str, help="Output directory for train.csv")
    parser.add_argument("--output-test-path", type=str, help="Output directory for test.csv")
    args = parser.parse_args()

    input_path = args.input_path or "/opt/ml/processing/input"
    output_train_path = args.output_train_path or "/opt/ml/processing/train"
    output_test_path = args.output_test_path or "/opt/ml/processing/test"
    print(f"--- Starting Processing Job ---")
    print(f"Input path: {input_path}")
    print(f"Output train path: {output_train_path}")
    print(f"Output test path: {output_test_path}")
    # Load the dataset
    print(f"Loading data from {input_path}/data.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    if not os.path.exists(os.path.join(input_path, "data.csv")):
        raise FileNotFoundError(f"Data file not found in {input_path}. Please check the path.")
    # Read the CSV file 
    data_path = os.path.join(input_path, "data.csv")
    df = pd.read_csv(data_path) 
    # Preprocess
    train, test = preprocessing(df)
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)
    print(f"Saving train data to {output_train_path}/train.csv")
    train.to_csv(os.path.join(output_train_path, "train.csv"), index=False)
    print(f"Saving test data to {output_test_path}/test.csv")
    test.to_csv(os.path.join(output_test_path, "test.csv"), index=False)
    print("--- Processing Job Completed ---")
