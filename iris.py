import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split

from fedot.api.main import Fedot

# Dataset
df = pd.read_csv("C:\\Users\\User\\Desktop\\Prog\\Datasets\\iris_dataset.csv")

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Fit and predict
model = Fedot(problem='classification',
              metric='accuracy',
              timeout=5,
              preset='best_quality',
              n_jobs=-1,
              seed=42,
              with_tuning=False,
            )

model.fit(features=X_train, target=y_train)

pipeline = model.current_pipeline
pipeline.show()

prediction = model.predict(features=X_test, save_predictions=True)

predictions_ = pd.read_csv("predictions.csv")
predictions_ = predictions_.applymap(lambda x: x.replace('[', '').replace(']', '').replace("'", '') if isinstance(x, str) else x)

sub = pd.DataFrame({"target": predictions_["Prediction"]})
sub['ID'] = sub.index
sub = sub[['ID', 'target']]
sub.to_csv("submission_copy_iris.csv", index=False)
