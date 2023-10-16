import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


data = pd.read_csv("Project 1 Data.csv")

grouped_data = data.groupby("Step")

group_stats = grouped_data.agg(['mean', 'std', 'min', 'max', 'count'])

# Display the statistics
print(group_stats)

# Create line plots for X-coordinate within each class
plt.figure(figsize=(12, 6))
for step, group in grouped_data:
    plt.plot(group['X'], group.index, label=f'Step {step}')

plt.title('X-coordinate Behaviour within Each Class')
plt.xlabel('Data Index')
plt.ylabel('X-coordinate')
plt.legend()
plt.show()

# Create line plots for Y-coordinate within each class
plt.figure(figsize=(12, 6))
for step, group in grouped_data:
    plt.scatter(group['Y'], group.index, label=f'Step {step}')

plt.title('Y-coordinate Behaviour within Each Class')
plt.xlabel('Data Index')
plt.ylabel('Y-coordinate')
plt.legend()
plt.show()

# Create line plots for Z-coordinate within each class
plt.figure(figsize=(12, 6))
for step, group in grouped_data:
    plt.scatter(group['Z'], group.index, label=f'Step {step}')

plt.title('Z-coordinate Behaviour within Each Class')
plt.xlabel('Data Index')
plt.ylabel('Z-coordinate')
plt.legend()
plt.show()

# For 3D plot for all steps
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data['X']
y = data['Y']
z = data['Z']
step = data['Step']

ax.scatter(x, y, z, c=step, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.colorbar(ax.scatter(x, y, z, c=step, cmap='viridis'))

plt.show()

# Correlation matrix
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix["Step"]

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

X = data.drop(columns=["Step"])  # Features
y = data["Step"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()

# Defining hyperparameters and their possible values to search through
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Assess the best Random Forest model's performance on the test set
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

svm_classifier = SVC()

# Defining hyperparameters and their possible values to search through
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] + [0.1, 1]
}

grid_search_svm = GridSearchCV(svm_classifier, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
best_svm_model = grid_search_svm.best_estimator_

# Assess the best SVM model's performance on the test set
y_pred_svm = best_svm_model.predict(X_test)
print("Support Vector Machine (SVM):")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

knn_classifier = KNeighborsClassifier()

# Defining hyperparameters and their possible values to search through
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

grid_search_knn = GridSearchCV(knn_classifier, param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
best_knn_model = grid_search_knn.best_estimator_

# Assess the best KNN model's performance on the test set
y_pred_knn = best_knn_model.predict(X_test)
print("K-Nearest Neighbors (KNN):")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Confusion matrix from model's predictions and true labels
conf_matrix = confusion_matrix(y_test, y_pred_svm)

# Heatmap for visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Save the selected model to a file
joblib.dump(best_svm_model, 'best_svm_model.joblib')
# Load the saved model
loaded_model = joblib.load('best_svm_model.joblib')

# Predict maintenance steps for the given coordinates
coordinates = [[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]]

predictions = loaded_model.predict(coordinates)

# The 'predictions' variable now contains the predicted maintenance steps for the given coordinates
