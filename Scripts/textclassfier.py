from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


class TextClassifier:

    def __init__(self, model, vectorizer=None):
        self.model = model
        self.vectorizer = vectorizer
        self.X_train_tf = None
        self.X_test_tf = None

    def train(self, X_train, y_train):
        self.X_train_tf = self.vectorizer.fit_transform(X_train)
        self.model.fit(self.X_train_tf, y_train)

    def evaluate(self, X_test, y_test):
        self.X_test_tf = self.vectorizer.transform(X_test)

        y_pred = self.model.predict(self.X_test_tf)
        print(f"f1_score:{f1_score(y_test, y_pred, average='macro')}")
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Accuracy: {acc}")
        print(report)

    def predict(self, text):
        text_tf = self.vectorizer.transform([text])
        return self.model.predict(text_tf)[0]
