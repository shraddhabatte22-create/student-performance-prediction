import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

print("Student Performance Prediction Project")
print("Libraries Imported Successfully!")
# Create Student Dataset

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 1, 3],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95, 50, 68],
    'Previous_Marks': [40, 45, 50, 55, 60, 65, 70, 75, 30, 48],
    'Final_Marks': [42, 48, 55, 60, 68, 75, 82, 90, 35, 50]
}

df = pd.DataFrame(data)

print("\nStudent Dataset:")
print(df)
# Separate Input and Output

X = df[['Study_Hours', 'Attendance', 'Previous_Marks']]
y = df['Final_Marks']

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))
# Create Machine Learning Model

model = LinearRegression()

# Train the Model

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")
# Predict Student Marks

y_pred = model.predict(X_test)

print("\nActual Marks:", list(y_test))
print("Predicted Marks:", y_pred.round(2))

# Calculate Error

error = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error:", round(error, 2))
# Predict Marks for New Student

new_student = pd.DataFrame({
    'Study_Hours': [6],
    'Attendance': [85],
    'Previous_Marks': [70]
})

predicted_marks = model.predict(new_student)

print("\nNew Student Prediction:")
print("Study Hours: 6")
print("Attendance: 85")
print("Previous Marks: 70")

print("Predicted Final Marks:", round(predicted_marks[0], 2))
# Student Performance Graph

plt.figure(figsize=(8, 5))

plt.bar(
    ['Actual Marks', 'Predicted Marks'],
    [y_test.iloc[0], y_pred[0]]
)

plt.title("Student Performance Prediction")
plt.xlabel("Marks Type")
plt.ylabel("Marks")

plt.show()
# Graph 1: Study Hours vs Final Marks

plt.figure(figsize=(8,5))

plt.scatter(df['Study_Hours'], df['Final_Marks'])

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")

plt.grid()
plt.show()


# Graph 2: Actual vs Predicted Marks

plt.figure(figsize=(8,5))

plt.plot(list(y_test), label="Actual Marks", marker='o')
plt.plot(y_pred, label="Predicted Marks", marker='o')

plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Actual vs Predicted Marks")

plt.legend()
plt.grid()
plt.show()
# Student Input System

print("\nEnter Student Details")

study = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance: "))
previous = float(input("Enter Previous Marks: "))

new_student = pd.DataFrame({
    'Study_Hours': [study],
    'Attendance': [attendance],
    'Previous_Marks': [previous]
})

prediction = model.predict(new_student)

print("\nStudent Performance Result")
print("Predicted Final Marks:", round(prediction[0], 2))