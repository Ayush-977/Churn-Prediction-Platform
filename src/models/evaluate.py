from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(
        model,
        X_test,
        y_test
):

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:,1]

    return {
        "accuracy":
        accuracy_score(y_test,pred),

        "precision":
        precision_score(y_test,pred),

        "recall":
        recall_score(y_test,pred),

        "f1":
        f1_score(y_test,pred),

        "roc_auc":
        roc_auc_score(y_test,prob)
    }
