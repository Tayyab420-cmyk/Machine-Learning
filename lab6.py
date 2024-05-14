import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Read data from CSV file
file_path = "test.csv"  # Assuming the file is named test.csv and is in the same directory
df = pd.read_csv(file_path)

# Step 2: Prepare data
X = df.drop(columns=["Gender"])  # Adjust target_column_name to the name of your output/label column
y = df["Gender"]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the model
lr_model = LogisticRegression(max_iter=1000)  # Adjust max_iter as needed
lr_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = lr_model.predict(X_test_scaled)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))