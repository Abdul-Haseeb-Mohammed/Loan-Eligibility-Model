import pandas as pd
from src.visualization.visualize import LoanStatus_barchart, LoanAmount_displot

def load_data(data_path):
    df = pd.read_csv(data_path)
    
    print("Printing first 5 rows:")
    print(df.head())
    
    print("Shape of dataset is:",df.shape)
    
    print("Check dtypes of dataset:")
    print(df.dtypes)
    
    return df

def clean_dataset(dataset):
    
    print("How many application were approved and how many were denied?")
    LoanStatus_barchart(dataset)
    
    print("Check for missing values:")
    print(dataset.isnull().sum())
    
    print("Check dtypes of columns:")
    print(dataset.dtypes)
    

    print("Distribution of Loan Amount:")
    LoanAmount_displot(dataset)
    
    dataset['Gender'].fillna('Male', inplace=True)
    dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
    dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
    dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace=True)

    dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
    dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
    dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)
    
    print("Confirm if there are any missing values left:")
    print(dataset.isnull().sum())
    
    dataset.drop('Loan_ID', axis=1,inplace=True)
    dataset = pd.get_dummies(dataset, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'],dtype=int)
    
    print("Preview processed dataset")
    print(dataset.head())
    
    dataset.to_csv("Loan-Eligibiltity-Model/data/processed",index=None)
    
    return dataset