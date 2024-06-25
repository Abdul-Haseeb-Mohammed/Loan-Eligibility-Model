from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
# Function to train the model
def train_logistic_regression(x_train_scaled, y_train):
    lrmodel = LogisticRegression().fit(x_train_scaled, y_train)
    return lrmodel

def train_decision_tree_classifier(X_train, y_train):
    dtc_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf= 5,max_features='sqrt', random_state=567).fit(X_train,y_train)
    return dtc_model

def train_random_forest_classifier(X_train_scaled, y_train):
    rfc_model = RandomForestClassifier(n_estimators=100, 
                                 min_samples_leaf=5, max_depth=5, 
                                 max_features='sqrt').fit(X_train_scaled, y_train)
    return rfc_model

#Funcation to pick the best model using cross validation
def kfold_model_selection(model_name, x_scaled, y, fold):
    if model_name.lower() == "logistic regression":
        model = LogisticRegression()
    elif model_name.lower() == "decision tree classifier":
        model = DecisionTreeClassifier()
    elif model_name.lower() == "random forest classifier":
        model= RandomForestClassifier()
    # Set up KFold cross-validation
    kfold = KFold(n_splits=fold, shuffle=True, random_state=123)

    # Initialize variables to track the best model
    best_score = -1
    best_model = None
    best_fold = None
    
    scores = []
    for fold, (train_index, test_index) in enumerate(kfold.split(x_scaled, y)):
        # Split the data into training and testing sets for the current fold
        X_train_fold, X_test_fold = x_scaled[train_index], x_scaled[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        # Train the model
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate the model
        score = model.score(X_test_fold, y_test_fold)
        scores.append(score)

        # Print the performance of each fold
        print(f"Fold {fold+1}: Accuracy = {score}")
        
        # Check if this model is the best so far
        if score > best_score:
            best_score = score
            best_model = model  # Save the best model
            best_fold = fold + 1  # Save the fold number

    scores = np.array(scores)

    # Print the accuracy scores for each fold
    print("Accuracy scores:", scores)

    # Print the mean accuracy and standard deviation of the model
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())

    # Print the best model's information
    print(f"\nBest Model: Fold {best_fold}, Accuracy = {best_score}")
    return best_model
