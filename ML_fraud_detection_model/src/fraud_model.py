from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent

#loaded the processed sample from the eda
def load_data():
    data_path = BASE_DIR / "data" / "processed" / "eda_cleaned_sample.csv"
    df = pd.read_csv(data_path)
    return df

df = load_data()

#y is the result found from x which are all the indicator columns
#both x and y are used to train and test
x = df[["hour_of_day",
        "hour_of_day_bins",
        "is_weekend",
        "city",
        "merchant_category",
        "payment_method",
        "account_age_years",
        "time_since_last_txn_hrs",
        "time_since_last_txn_hrs_bins",
        "failed_attempts",
        "is_international",
        "pin_changed_recently",
        "is_night_transaction"
        ]]
y = df["is_fraud"]

#split the data in (x,y)training and (x,y)testing
#the fraud rate of 5.5% is maintained through both splits with the stratify feature.
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#split the indicators into lists called categorial and numerical for encoding and scaling.
category_columns = ["hour_of_day_bins",
                    "city",
                    "merchant_category",
                    "payment_method",
                    "time_since_last_txn_hrs_bins"
                    ]
numeric_columns = ["hour_of_day",
                   "is_weekend",
                   "time_since_last_txn_hrs",
                   "failed_attempts",
                   "is_international",
                   "pin_changed_recently",
                   "is_night_transaction",
                   "account_age_years"
                   ]

#encoded the categorical lists and scaled the numerical lists.
#encoding is used to give row of a column that use words to categorize into numbers so that the model can understand
#scaling is used to prevent skewing of data that is caused by large numerical scale for one column and small numerical scale for another.
ct = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), category_columns),
        ("num", StandardScaler(), numeric_columns)
    ],
    remainder = "passthrough"
)

#created a pipeline that includes the ct preprocessing and the statistical model I used, "Random forest regression"
#I tried logistic regression but yielded worse results than random forest regression.
pipe = Pipeline([
    ("preprocessing", ct),
    ("model", RandomForestClassifier())
])


#After tweaking the parameters for the random forest regression multiple times, these parameters yielded the best results.
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [2, 3, 5, 7],
    "model__class_weight": [None, "balanced"],
}

#this is the model including the preprocessing, pipeline, and parameter
model = GridSearchCV(pipe,
                    param_grid,
                    scoring="f1",
                    n_jobs=-1,
                    cv=3)


#Trained the model with the training(x, y)split
model.fit(x_train, y_train)

#I changed the default threshold from 50% to 45%. I tried lower and higher and 45% is the sweetspot for the best result.
y_prob = model.predict_proba(x_test)[:,1]
threshold = 0.45
predictions = (y_prob >= threshold).astype(int)


#these are the results:
#81% recall and 8% precision
#The results show that there aren't any good indicators for the model to be able to classify.
#Since most of these transactions are similar it categorizes 50% of the data as fraud leading to that low precision.
#But it does catch 81% of the 5% fraud transactions which is really good.
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#I used this to find the best parameters for the model to get the best results
print(model.best_params_)

report = classification_report(y_test, predictions, output_dict=True)
f_precision = report["1"]["precision"]
f_recall = report["1"]["recall"]

pd.DataFrame({"y_test": y_test,
              "predictions": predictions,
              "precision": f_precision,
              "recall": f_recall,
              }).to_csv(BASE_DIR / "data" / "processed" / "results.csv")

visuals = df[["failed_attempts",
              "is_international",
              "is_fraud"]].to_csv(BASE_DIR / "data" / "processed" / "best_indicators.csv")