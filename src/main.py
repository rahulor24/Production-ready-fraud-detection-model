from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor

# load and split data

from sklearn.model_selection import train_test_split


class Classifier:
    def data(df):
        if 'fraud_flag' in df.columns:
            X = df.drop('fraud_flag', axis=1)
            y = df['fraud_flag']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            return X_train, X_test, y_train, y_test
        else:
            X_test = df
        return X_test

    def RandForest(X_train, y_train):
        rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_depth=10,
                                    random_state=42, class_weight='balanced', n_jobs=-1, oob_score=True)

        rfp = Pipeline([
            ('preprocessor', Preprocessor()),
            ('model', rf)
        ])

        rfp.fit(X_train, y_train)
        return rf, rfp

    def prediction(model, X):
        y_prob = model.predict_proba(X)[:,1]
        y_pred = (y_prob >= 0.36).astype(int)     # best threshold = 0.36
        return y_pred, y_prob

    def metrics(Y, y_pred, y_prob):
        acc = accuracy_score(Y, y_pred)
        f1 = f1_score(Y, y_pred)
        pr = precision_score(Y, y_pred)
        rec =recall_score(Y, y_pred)
        roc = roc_auc_score(Y,y_prob)
        cm = confusion_matrix(Y,y_pred)
        return acc, f1, pr, rec, roc, cm