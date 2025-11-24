import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve)
import joblib
import optuna
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load and preprocess data
df = pd.read_csv('dataset2.csv').drop('index', axis=1)

# Convert target to binary (0/1)
df['Result'] = df['Result'].replace({-1: 0, 1: 1})

# Split features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameter search space
    max_depth = trial.suggest_int('max_depth', 1, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    # Create a Decision Tree model with the suggested hyperparameters
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Optuna minimizes the objective, so we return 1 - accuracy
    return 1 - accuracy

# Create an Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # Adjust number of trials as needed

# Get the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
best_model = DecisionTreeClassifier(
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)  # Calculate ROC-AUC score

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")  # Print ROC-AUC score

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save the best model
joblib.dump(best_model, 'optimized_decision_tree_model.pkl')
print("Optimized Decision Tree model saved as 'optimized_decision_tree_model.pkl'")

# Feature importance
feature_importance = best_model.feature_importances_
feature_importance_dict = dict(zip(X.columns, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nTop 3 most important features:")
for feature, importance in sorted_features[:3]:
    print(f"{feature}: {importance:.4f}")