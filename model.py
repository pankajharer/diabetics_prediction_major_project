import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Importing Dataset
dataset = pd.read_csv('diabetes-dataset.csv')

# Data Preprocessing
dataset_X = dataset.iloc[:, [1, 4, 5, 7]].values
dataset_Y = dataset.iloc[:, 8].values

sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)
X = pd.DataFrame(dataset_scaled)
Y = dataset_Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])

# Data Modelling
dt = DecisionTreeClassifier(random_state=42)
ada_bias_dt_1 = AdaBoostClassifier(estimator=dt, n_estimators=10, algorithm='SAMME', random_state=42)
sample_weights = [0.7 if i == 1 else 0.4 for i in y_train]
ada_bias_dt_1.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluating the Model
y_pred = ada_bias_dt_1.predict(X_test)
accuracy_bias_dt_1 = accuracy_score(y_test, y_pred)
f1_bias_dt_1 = f1_score(y_test, y_pred)

print("Accuracy of adaboost bias weights dt:", round(accuracy_bias_dt_1, 3) * 100, "%")
print("F1-score:", round(f1_bias_dt_1, 2))

# Saving the Model and Scaler
joblib.dump(ada_bias_dt_1, 'model.pkl')

# Keep script running
input("Press Enter to exit...")
