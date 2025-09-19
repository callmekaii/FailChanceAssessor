# FailChanceAssessor

This project uses a **feedforward neural network (Keras)** to predict whether a student will **pass or fail** based on their academic and personal data.  

## Dataset
The dataset (`student_performance_dataset.csv`) includes:
- `Gender(Coded)` → Encoded gender (e.g., 0 = Male, 1 = Female)  
- `Study_Hours` → Average hours studied per day  
- `Attendance_Rate` → Attendance percentage  
- `Past_Exam_Scores` → Average score from past exams  
- `Internet_Access_at_Home(Coded)` → Encoded (0 = No, 1 = Yes)  
- `Extracurricular_Activities(Coded)` → Encoded (0 = No, 1 = Yes)  
- `Final_Exam_Score` → Final exam raw score  
- `Pass_Fail(Coded)` → Target variable (0 = Fail, 1 = Pass)  

## Model Architecture
The model is a simple **Sequential Neural Network**:
- Input layer → matches number of features  
- Hidden Layer 1 → Dense(16, ReLU)  
- Hidden Layer 2 → Dense(8, ReLU)  
- Output Layer → Dense(1, Sigmoid)  

Loss: `binary_crossentropy`  
Optimizer: `adam`  
Metric: `accuracy`

## Training
- Dataset split: 80% training / 20% testing  
- StandardScaler applied to normalize features  
- 50 epochs with 20% validation split  

## Credits
- https://www.kaggle.com/datasets/amrmaree/student-performance-prediction
Got the dataset from here lul. Try going bananas there, y'all might be able to come up with smth new