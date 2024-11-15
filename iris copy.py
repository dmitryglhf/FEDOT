import numpy as np
import pandas as pd

from fedot.api.main import Fedot

# Dataset
train = pd.read_csv("C:\\Users\\User\\Desktop\\Prog\\Datasets\\iris_train.csv")
test = pd.read_csv("C:\\Users\\User\\Desktop\\Prog\\Datasets\\iris_test.csv")

# Fit and predict
model = Fedot(problem='classification',
              metric='accuracy',
              timeout=5,
              preset='best_quality',
              n_jobs=-1,
              seed=42,
              with_tuning=False,
            )

model.fit(features=train, target='target')

pipeline = model.current_pipeline
pipeline.show()

prediction = model.predict(features=test, save_predictions=True)

# predictions_ = pd.read_csv("predictions.csv")
# predictions_ = predictions_.applymap(lambda x: x.replace('[', '').replace(']', '').replace("'", '') if isinstance(x, str) else x)

# test_data = pd.read_csv("C:\\Users\\User\\Desktop\\Prog\\Datasets\\iris_test.csv")

# sub = pd.DataFrame({"target": predictions_["Prediction"]})
# sub['ID'] = sub.index
# sub = sub[['ID', 'target']]
# sub.to_csv("submission_copy.csv", index=False)
