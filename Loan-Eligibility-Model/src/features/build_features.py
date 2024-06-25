# import module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_test_splitter(x_scaled, Loan_Status):
    # Split the dataset using stratified sampling to ensure both classes are split in good proportion
    x_train, x_test, y_train, y_test = train_test_split(x_scaled,Loan_Status, test_size=0.2, random_state=123, stratify=Loan_Status)
    
    print("Train test split shape: ")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test

def standardize(x):
    scale = StandardScaler()
    return scale.fit_transform(x)