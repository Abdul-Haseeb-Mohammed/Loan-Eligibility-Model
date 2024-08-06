import pickle
import warnings
warnings.filterwarnings("ignore")
from src.data.make_dataset import load_data, clean_dataset
from src.features.build_features import train_test_splitter, standardize
from src.models.train_model import train_logistic_regression, train_decision_tree_classifier, train_random_forest_classifier, kfold_model_selection
from src.models.predict_model import run_and_evaluate_model, run_and_evaluate_model_logistic_regression
from src.visualization.visualize import plot_tree

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "Loan-Eligibility-Model/data/raw/credit.csv"
    df = load_data(data_path)

    #cleaning the data
    cleaned_dataset = clean_dataset(df.copy())
    
    #Using standardization to scale the data
    x_scaled = standardize(cleaned_dataset.drop('Loan_Status',axis=1))
    
    # Split the dataset into train test
    X_train, X_test, y_train, y_test = train_test_splitter(x_scaled, cleaned_dataset['Loan_Status'])
    
    #Train the Linear regression model
    lrmodel = train_logistic_regression(X_train, y_train)

    # Save the trained model
    with open('Loan-Eligibility-Model/models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lrmodel, f)

    # Show the metrics of models
    print("Coefficients of Logistic Regression:", lrmodel.coef_)
    print("Intercept of Logistic Regression:", lrmodel.intercept_)
    
    # Display evaluation metrics for logistic regression
    lr_train_mae, lr_test_mae, lr_train_cf, lr_test_cf = run_and_evaluate_model_logistic_regression(lrmodel, X_train, X_test, y_train, y_test)
    print('Logistic Regression Train error is', lr_train_mae)
    print('Logistic Regression Test error is', lr_test_mae)
    print("Logistic Regression Train Confusion Matrix:", lr_train_cf)
    print("Logistic Regression Test Confusion Matrix:", lr_test_cf)
    
    #Train the Decision Tree Classifier model
    dtc_model = train_decision_tree_classifier(X_train, y_train)
    
    # Save the trained model
    with open('Loan-Eligibility-Model/models/decision_tree_classifier.pkl', 'wb') as f:
        pickle.dump(dtc_model, f)

    # Display evaluation metrics for Decision Tree Classifier
    dtc_train_mae, dtc_test_mae, dtc_train_cf, dtc_test_cf = run_and_evaluate_model(dtc_model, X_train, X_test, y_train, y_test)
    print('Decision Tree Classifier Train error is', dtc_train_mae)
    print('Decision Tree Classifier Test error is', dtc_test_mae)
    print("Decision Tree Classifier Confusion Matrix:", dtc_train_cf)
    print("Decision Tree Classifier Confusion Matrix:", dtc_test_cf)
    
    #Plot the decision tree
    #plot_tree(dtc_model, dtc_model.feature_names_in_, save_path='Loan-Eligibility-Model/reports/figures/decision_tree.png')
    
    #Train the Random Forest Classifier
    rfc_model = train_random_forest_classifier(X_train, y_train)
    
    # Save the trained model
    with open('Loan-Eligibility-Model/models/random_forest_classifier.pkl', 'wb') as f:
        pickle.dump(rfc_model, f)
 
    # Display evaluation metrics for Random Forest Classifier
    rfc_train_mae, rfc_test_mae, rfc_train_cf, rfc_test_cf = run_and_evaluate_model(rfc_model, X_train, X_test, y_train, y_test)
    print('Random Forest Regressor Train error is', rfc_train_mae)
    print('Random Forest Regressor Test error is', rfc_test_mae)
    print("Random Forest Regressor Confusion Matrix:", rfc_train_cf)
    print("Random Forest Regressor Confusion Matrix:", rfc_test_cf)
    
    #Plot the Random Forest Regressor
    #plot_tree(rfc_model.estimators_[2], dtc_model.feature_names_in_, save_path='Loan-Eligibility-Model/reports/figures/random_forst_decision_tree2.png')
    
    print("Using 5-Fold Cross Validation to pick best model:")
    lr_best_model = kfold_model_selection("Logistic Regression", x_scaled, cleaned_dataset['Loan_Status'],5)
    
    # Save the best model to a file
    best_logistic_regression_model_filename = 'best_model_logistic_regression.pkl'
    with open("Loan-Eligibility-Model/models/" + best_logistic_regression_model_filename, 'wb') as file:
        pickle.dump(lr_best_model, file)
    print(f"Saved logistic regression best model as '{best_logistic_regression_model_filename}'")
    
    dtc_best_model = kfold_model_selection("Decision Tree Classifier", x_scaled, cleaned_dataset['Loan_Status'],5)
    
    # Save the best model to a file
    best_decision_tree_classifier_filename = 'best_model_Decision_Tree_Classifier.pkl'
    with open("Loan-Eligibility-Model/models/" + best_decision_tree_classifier_filename, 'wb') as file:
        pickle.dump(dtc_best_model, file)
    print(f"Saved Decision Tree Classifier best model as '{best_decision_tree_classifier_filename}'")
    
    rfc_best_model = kfold_model_selection("Random Forest Classifier", x_scaled, cleaned_dataset['Loan_Status'],5)
    
    # Save the best model to a file
    best_random_forest_classifier_filename = 'best_model_Random_Forest_Classifier.pkl'
    with open("Loan-Eligibility-Model/models/" + best_random_forest_classifier_filename, 'wb') as file:
        pickle.dump(rfc_best_model, file)
    print(f"Saved Random Forest Classifier best model as '{best_random_forest_classifier_filename}'")
    
    
    