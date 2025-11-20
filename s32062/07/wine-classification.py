import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

columns = [
    "Class",
    "Alcohol",
    "Malic_acid",
    "Ash",
    "Alcalinity_of_ash",
    "Magnesium",
    "Total_phenols",
    "Flavanoids",
    "Nonflavanoid_phenols",
    "Proanthocyanins",
    "Color_intensity",
    "Hue",
    "OD280/OD315",
    "Proline"
]

df = pd.read_csv(url, header=None, names=columns)
# df.head()
# df.info()
# df.describe()
# df['Class'].value_counts()

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

rf = RandomForestClassifier(random_state=1)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)

print(report)

rf_file_path = 'random-forest-wine.pkl'
joblib.dump(rf, rf_file_path)
rf_size = os.path.getsize(rf_file_path)
print(rf_size)
