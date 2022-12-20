import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

titanic_df = pd.read_csv('https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv')
st.header("Titanic Project")
st.subheader("predicting death")

st.sidebar.subheader("Settings")
if st.sidebar.checkbox("Display DataFrame"):
    st.write("The DataFrame")
    #st.write(titanic_df)
if st.button("same thing"):
    st.write("yup, it works")

st.subheader("Plots")
col_1, col_2 = st.columns(2)

with col_1:
    fig, ax = plt.subplots()
    titanic_df.Age.hist(ax=ax)
    st.write(fig)
    st.caption("Age distribution")
with col_2:
    fig, ax = plt.subplots()
    titanic_df.Sex.hist(ax=ax)
    st.write(fig)
    st.caption("Sex distribution")


