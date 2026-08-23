from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent

def load_data():
    data_path = BASE_DIR / "data" / "processed" / "results.csv"
    df = pd.read_csv(data_path)
    return df

def load_data1():
    data_path = BASE_DIR / "data" / "processed" / "best_indicators.csv"
    df = pd.read_csv(data_path)
    return df

results = load_data()
indicators = load_data1()



#1st Graph: 2x2 Confusion matrix
plt.figure()
cm = confusion_matrix(results["y_test"], results["predictions"])
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap= "Blues",
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"],
            )
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")

plt.savefig(BASE_DIR / "Visualizations" / "confusion_matrix.png")



#2nd Graph: Precision vs Recall Bar Chart
plt.figure()

name = np.array(["Precision", "Recall"])
rate = np.array([results["precision"][0] * 100, results["recall"][0] * 100])
scores = [round(results["precision"][0] * 100, 1), round(results["recall"][0] * 100, 1)]
plt.ylabel("Results In %")
plt.ylim(0, 100)
plt.title("Precision vs Recall")

for i, score in enumerate(scores):
    plt.text(i, score + 2, f"{score}%", ha="center")

plt.bar(name, rate, color=["#6BAED6", "#08519C"], edgecolor="black")
plt.savefig(BASE_DIR / "Visualizations" / "Precision.V.Recall.png")



#3rd Graph: Fraud rate by International vs. Non_International transactions
plt.figure()

mean_1 = indicators.groupby("is_international")["is_fraud"].mean()

name = ["Non-International", "International"]
values = []
for i, value in mean_1.items():
    values.append(round(value * 100, 1))

for i, score in enumerate(values):
    plt.text(i, score + 2, f"{score}%", ha="center")

plt.title("International VS. Non-International")
plt.ylabel("Fraud Rate In %")
plt.ylim(0, 100)

plt.bar(name, values, color=["#6BAED6", "#08519C"], edgecolor="black")
plt.savefig(BASE_DIR / "Visualizations" / "International.v.Non-International.png")



#4th Graph: Fraud rate by failed attempts
plt.figure()

mean_2 = indicators.groupby(["failed_attempts"])["is_fraud"].mean()

name1 = []
values1 = []
for i, value in mean_2.items():
    name1.append(i)
    values1.append(round(value * 100, 1))

for i, score in enumerate(values1):
    plt.text(i, score + 2, f"{score}%", ha="center")

plt.title("Fraud Rate By Failed Attempts")
plt.xlabel("# of Failed Attempts")
plt.ylabel("Fraud Rate In %")
plt.ylim(0, 100)

plt.bar(name1, values1, color="#6BAED6", edgecolor="black")
plt.savefig(BASE_DIR / "Visualizations" / "failed_attempts.png")



