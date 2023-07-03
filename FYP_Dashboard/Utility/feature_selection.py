import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


class FeatureSelection:
    df = pd.DataFrame()
    className = ""
    X, y = [], []
    X_train_var, X_test_var, yTrain, yTest, x_train_adv, x_test_adv = [], [], [], [], [], []

    # Step 4: Set up a candidate feature selection method pool
    feature_selection_methods = [
        ('Univariate Selection', SelectKBest(score_func=mutual_info_classif, k=10)),
        ('Univariate Selection (ANOVA F-value)', SelectKBest(score_func=f_classif, k=10)),
        ('Random Forest Importance', RandomForestClassifier(n_estimators=100)),
        ('Wrapper Method (Recursive Feature Elimination)', RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=10)),
        # Add more feature selection methods as desired
    ]

    # Step 5: Implement a loop or search algorithm
    results = {}  # Dictionary to store performance results for each method

    def split_data(self):
        self.X = self.df.drop(self.className, axis=1)
        self.y = self.df[self.className]

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=0)

    def evaluate_methods(self):

        global X_train_selected
        print("evaluate_methods call")
        for method_name, method in self.feature_selection_methods:
            # Apply feature selection method
            # Preprocess the data to ensure non-negativity
            print("start evaluate_method", method_name)

            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_val_scaled = scaler.transform(self.X_val)

            if isinstance(method, (SelectKBest, RFE)):
                X_train_selected = method.fit_transform(X_train_scaled, self.y_train)
                X_val_selected = method.transform(X_val_scaled)
            elif isinstance(method, SelectKBest):
                X_train_selected = method.fit_transform(self.X_train, self.y_train)
                X_val_selected = method.transform(self.X_val)
            elif isinstance(method, RandomForestClassifier):
                method.fit(X_train_scaled, self.y_train)
                importances = method.feature_importances_
                indices = np.argsort(importances)[::-1]
                X_train_selected = X_train_scaled[:, indices[:10]]
                X_val_selected = X_val_scaled[:, indices[:10]]

            # Train a classifier and make predictions
            print("start training.... ")

            clf = KNeighborsClassifier(n_neighbors=100)
            clf.fit(X_train_selected, self.y_train)
            y_pred = clf.predict(X_val_selected)

            # Evaluate performance
            performance = accuracy_score(self.y_val, y_pred)
            print(method_name, performance)

            # Store results
            self.results[method_name] = performance
            print("End")

    def find_best_method(self):
        # Step 6: Compare and rank results
        ranked_methods = sorted(self.results.items(), key=lambda x: x[1], reverse=True)

        # Step 7: Select the best method
        best_method = ranked_methods[0][0]
        print("Selected method:", best_method)

        # Step 8: Validate and fine-tune (using the best method)
        # Apply the best method on the entire dataset
        best_method_obj = []
        for method_name, method in self.feature_selection_methods:
            if method_name == best_method:
                if isinstance(method,  (SelectKBest, RFE)):
                    best_method_obj = method.get_support(indices=True)
                elif isinstance(method, RandomForestClassifier):
                    importance = method.feature_importances_
                    indices = np.argsort(importance)[::-1]
                    best_method_obj = indices[:10]
                break
        # Transform the entire dataset using the best method
        X_selected = self.X.iloc[:, best_method_obj]

        X_fe_train, X_fe_test, y_fe_train, y_fe_test = train_test_split(X_selected, self.y, test_size=0.2, random_state=42)

        return X_fe_train, X_fe_test, y_fe_train, y_fe_test, best_method