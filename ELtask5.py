# Ask 5: Decision Trees and Random Forests

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load Dataset
df = pd.read_csv(r"C:\Users\hemap\Downloads\heart.csv")   # Change path if needed

# 3. Separate Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# STEP 1: Train Decision Tree Classifier
# ---------------------------------------------------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", round(dt_accuracy, 4))

# Visualize Decision Tree
plt.figure(figsize=(18,10))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree")
plt.show()

# ---------------------------------------------------
# STEP 2: Analyze Overfitting by Controlling Tree Depth
# ---------------------------------------------------
depths = range(1, 11)
train_acc = []
test_acc = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)

    train_acc.append(model.score(X_train, y_train))
    test_acc.append(model.score(X_test, y_test))

plt.figure(figsize=(8,5))
plt.plot(depths, train_acc, marker='o', label="Training Accuracy")
plt.plot(depths, test_acc, marker='o', label="Testing Accuracy")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------
# STEP 3: Train Random Forest and Compare Accuracy
# ---------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", round(rf_accuracy,4))

print("\nAccuracy Comparison")
print("--------------------------")
print("Decision Tree :", round(dt_accuracy,4))
print("Random Forest :", round(rf_accuracy,4))

# ---------------------------------------------------
# STEP 4: Feature Importances
# ---------------------------------------------------
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importances")
print(importance)

plt.figure(figsize=(8,5))
importance.plot(kind="bar")
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance Score")
plt.show()

# ---------------------------------------------------
# STEP 5: Evaluate using Cross Validation
# ---------------------------------------------------
cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Accuracy for each Fold:")
print(cv_scores)

print("\nAverage Cross Validation Accuracy:",
      round(cv_scores.mean(),4))
