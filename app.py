import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import joblib
st.title("2025 DSN Artificial 🚀intelligence Bootcamp")
st.image("DSN.JPG")
st.divider()
st.header("African Youth Unemployment Distribution")
path="youth_unemployment_africa.csv"
dataset=pd.read_csv(path)
st.write(dataset)
if st.button("summarize"):
    st.write(dataset.describe())
if st.button("Show Relation"):
    # Quick scatter: GDP vs Youth Unemployment
    ax= plt.figure()
    plt.scatter(dataset['GDP_Per_Capita'], dataset['Youth_Unemployment_Rate'])
    plt.title('Youth Unemployment vs GDP per Capita')
    plt.xlabel('GDP per Capita (USD)')
    plt.ylabel('Youth Unemployment Rate (%)')
    plt.show()
    st.pyplot(ax)
    # Quick scatter: Education Index vs Youth Unemployment
    ay=plt.figure()
    plt.scatter(dataset['Education_Index'], dataset['Youth_Unemployment_Rate'])
    plt.title('Youth Unemployment vs Education Index')
    plt.xlabel('Education Index')
    plt.ylabel('Youth Unemployment Rate (%)')
    plt.show()
    st.pyplot(ay)
#loading the model 
model=joblib.load("model.pkl")
st.divider()
st.write("Enter the number of features here")
value1=st.number_input("Enter GDP per Capital")
value2=st.number_input("Enter Education Index")
value3=st.number_input("Enter Urban_Population_Percent")
predictions= model.predict([[value1,value2,value3]])
if st.button("predict"):
    st.write(predictions)
    st.balloons()
st.divider()
st.write("DSN is amazing")