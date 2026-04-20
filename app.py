import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv")

st.title("Diabetes Prediction System")
st.sidebar.header("Enter Patient Data")

st.subheader("Dataset Summary")
st.write(df.describe())

X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 25)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.5)
    age = st.sidebar.slider('Age', 21, 88, 33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    return pd.DataFrame(data, index=[0])

user_data = user_input()

st.subheader("Patient Input Data")
st.write(user_data)

model = RandomForestClassifier()
model.fit(X_train, y_train)

prediction = model.predict(user_data)
accuracy = accuracy_score(y_test, model.predict(X_test))

st.subheader("Prediction Result")
if prediction[0] == 0:
    st.success("✅ You are NOT Diabetic")
    color = "blue"
else:
    st.error("⚠️ You are Diabetic")
    color = "red"

st.write(f"Model Accuracy: {accuracy*100:.2f}%")

st.header("Data Visualizations")

st.subheader("Age Distribution")
fig1 = plt.figure()
sns.histplot(df['Age'], kde=True)
st.pyplot(fig1)

st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)

st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

st.subheader("Correlation Heatmap")
fig4 = plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot(fig4)

st.subheader("Age vs BMI (User vs Dataset)")
fig5 = plt.figure()
sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome')
plt.scatter(user_data['Age'], user_data['BMI'], color=color, s=200)
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig5)

st.subheader("Age vs Glucose Trend")
fig6 = plt.figure()
sns.lineplot(x='Age', y='Glucose', data=df,hue='Outcome')
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig6)