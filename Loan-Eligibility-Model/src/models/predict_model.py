# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix

def run_and_evaluate_model_logistic_regression(model, X_train, X_test, y_train, y_test, *threshold):
    if threshold:
        train_pred_prob = model.predict_proba(X_train)[:,1]
        train_pred = (train_pred_prob >= threshold).astype(int)
        train_mae = accuracy_score(y_train, train_pred)
        train_confusion_matrix = confusion_matrix(y_train, train_pred)
        
        test_pred_prob = model.predict_proba(X_test)[:,1]
        test_pred = (test_pred_prob >= threshold).astype(int)
        test_mae = accuracy_score(y_test, test_pred)
        test_confusion_matrix = confusion_matrix(y_test, test_pred)

    else:    
        train_pred = model.predict(X_train)
        train_mae = accuracy_score(y_train, train_pred)
        train_confusion_matrix = confusion_matrix(y_train, train_pred)

        test_pred = model.predict(X_test)
        test_mae = accuracy_score(y_test, test_pred)
        test_confusion_matrix = confusion_matrix(y_test, test_pred)

    return train_mae, test_mae, train_confusion_matrix, test_confusion_matrix

def run_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    train_mae = accuracy_score(y_train, train_pred)
    train_confusion_matrix = confusion_matrix(y_train, train_pred)

    test_pred = model.predict(X_test)
    test_mae = accuracy_score(y_test, test_pred)
    test_confusion_matrix = confusion_matrix(y_test, test_pred)

    return train_mae, test_mae, train_confusion_matrix, test_confusion_matrix

def run_and_evaluate_model_logistic_regression1(model, X_train, X_test, y_train, y_test, *threshold):
    if threshold:
        train_pred_prob = model.predict_proba(X_train)[:,1]
        train_pred = (train_pred_prob >= threshold).astype(int)
        train_mae = accuracy_score(y_train, train_pred)
        train_confusion_matix = confusion_matrix(X_test, train_pred)
        
        test_pred_prob = model.predict_proba(y_train)[:,1]
        test_pred = (test_pred_prob >= threshold).astype(int)
        test_mae = accuracy_score(test_pred, y_test)
        test_confusion_matix = confusion_matrix(y_test, test_pred)

    else:    
        train_pred = model.predict(X_train)
        train_mae = accuracy_score(train_pred, X_test)
        train_confusion_matix = confusion_matrix(X_test, train_pred)

        test_pred = model.predict(y_train)
        test_mae = accuracy_score(test_pred, y_test)
        test_confusion_matix = confusion_matrix(y_test, test_pred)

    return train_mae, test_mae, train_confusion_matix, test_confusion_matix

def run_and_evaluate_model1(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    train_mae = accuracy_score(train_pred, X_test)
    train_confusion_matix = confusion_matrix(X_test, train_pred)

    test_pred = model.predict(y_train)
    test_mae = accuracy_score(test_pred, y_test)
    test_confusion_matix = confusion_matrix(y_test, test_pred)

    return train_mae, test_mae, train_confusion_matix, test_confusion_matix