import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

#Import dataset
df = pd.read_csv("student_performance_dataset.csv")
#Features
X = df[["Gender(Coded)", "Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores", "Internet_Access_at_Home(Coded)", "Extracurricular_Activities(Coded)", "Final_Exam_Score"]]
y = df[["Pass_Fail(Coded)"]]
#Splitting Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def Model():
        model = keras.Sequential([
        layers.Dense(16, activation="relu", input_shape = (X_train.shape[1],)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # dropout probability (0-1)
    ])
    
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",   # dropout yes/no
            metrics=["accuracy"]
        )
        return model

model = Model()
#Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")


# Example student: [Gender(Coded)=1, Study_Hours=5, Attendance_Rate=90,
# Past_Exam_Scores=70, Internet_Access_at_Home=1, Extracurricular=0, Final_Exam_Score=60]
new_student = np.array([[1, 5, 30, 30, 1, 0, 36]])

# Scale with same scaler used on training data
new_student_scaled = scaler.transform(new_student)

# Predict
prediction_prob = model.predict(new_student_scaled)[0][0]
prediction = int(prediction_prob > 0.5)

print(f"\nPredicted Probability of Passing: {prediction_prob:.2f}")
print("Prediction:", "Pass" if prediction == 1 else "Fail")


