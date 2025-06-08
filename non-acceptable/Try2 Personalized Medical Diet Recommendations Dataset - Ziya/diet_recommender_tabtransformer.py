import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings
warnings.filterwarnings('ignore')

# 1. لود کردن دیتاست
data = pd.read_csv('Personalized_Diet_Recommendations.csv')

# 2. تحلیل داده‌ها با matplotlib
plt.figure(figsize=(8, 6))
data['Recommended_Meal_Plan'].value_counts().plot(kind='bar')
plt.title('Distribution of Recommended Meal Plans')
plt.xlabel('Meal Plan')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
data.boxplot(column=['Age', 'BMI', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic'])
plt.title('Box Plot of Numeric Features')
plt.xticks(rotation=45)
plt.show()

# 3. انتخاب ویژگی‌ها و هدف
numeric_features = ['Age', 'BMI', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 
                    'Cholesterol_Level', 'Blood_Sugar_Level', 'Daily_Steps', 
                    'Exercise_Frequency', 'Sleep_Hours']
categorical_features = ['Gender', 'Genetic_Risk_Factor', 'Alcohol_Consumption', 'Smoking_Habit', 
                        'Chronic_Disease', 'Allergies', 'Dietary_Habits']
target = 'Recommended_Meal_Plan'

X = data[numeric_features + categorical_features]
y = data[target]

# 4. پیش‌پردازش عمیق
# پر کردن مقادیر گمشده
X = X.fillna({'Chronic_Disease': 'None', 'Allergies': 'None', 'Dietary_Habits': 'None'})
for column in X.columns:
    X[column] = X[column].replace('', 'None').astype(str)

# تبدیل مقادیر عددی و پر کردن با میانگین
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].fillna(X[col].mean())

# حذف پرت‌ها با IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(X[numeric_features])
X = X[outliers == 1]
y = y[outliers == 1]

# کدگذاری لیبل
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# پیش‌پردازش با Tfidf و OneHot
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# 5. تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# 6. ساخت مدل TabTransformer
tabnet = TabNetClassifier(
    n_d=8, n_a=8, n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, verbose=0
)

# 7. آموزش مدل
tabnet.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=100, patience=10,
    batch_size=256, virtual_batch_size=128
)

# 8. ارزیابی مدل
y_pred = tabnet.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 9. پیش‌بینی برای بیمار جدید
new_patient = pd.DataFrame({
    'Age': [40], 'BMI': [25], 'Blood_Pressure_Systolic': [120], 'Blood_Pressure_Diastolic': [80],
    'Cholesterol_Level': [200], 'Blood_Sugar_Level': [100], 'Daily_Steps': [8000],
    'Exercise_Frequency': [3], 'Sleep_Hours': [7], 'Gender': ['Male'],
    'Genetic_Risk_Factor': ['No'], 'Alcohol_Consumption': ['No'], 'Smoking_Habit': ['No'],
    'Chronic_Disease': ['None'], 'Allergies': ['None'], 'Dietary_Habits': ['Regular']
})

new_patient_processed = preprocessor.transform(new_patient)
predicted_diet = tabnet.predict(new_patient_processed)
predicted_diet_label = label_encoder.inverse_transform(predicted_diet)
print(f'Recommended diet for the new patient: {predicted_diet_label[0]}')