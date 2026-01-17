from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor

# load and split data

import pandas as pd
from sklearn.model_selection import train_test_split


def RandForest(df):
    X = df.drop('fraud_flag', axis=1)
    y = df['fraud_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_depth=10,
                                random_state=42, class_weight='balanced', n_jobs=-1, oob_score=True)

    rfp = Pipeline([
        ('preprocessor', Preprocessor()),
        ('model', rf)
    ])

    rfp.fit(X_train, y_train)

    y_pred = rfp.predict(X_train)
    train_f1_score = f1_score(y_train, y_pred)

    y_prob = rfp.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.36).astype(int)     # best threshold = 0.36
    test_f1_score = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test,y_pred)

    return train_f1_score, test_f1_score, cm


train_f1, test_f1, cm = RandForest()

print("Train F1:", train_f1)
print("Test F1:", test_f1)
print("Confusion Matrix:\n", cm)
