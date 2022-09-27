import numpy as np
import pickle
import pandas as pd
import streamlit as st
import joblib
import time


st.image("https://images.unsplash.com/photo-1591444539769-2518e73d1090?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1325&q=80", use_column_width= 'always')

# Title
st.title("Predicting MLB Pitcher Salaries")

# Subtitle
st.markdown("Enter the following statistics for a pitcher and \
                    get an estimated salary value.")

# Add sidebar
st.sidebar.markdown("## Predict the Salary of a MLB Pitcher!")
st.sidebar.image("http://cdn.shopify.com/s/files/1/0480/9470/7866/collections/ef26964ae31041325cd9672682c01534.jpg?v=1646869133", width = 200)
st.sidebar.markdown("How does a team determine how much to pay their players?")

st.sidebar.markdown("#### Built by Eric Au")

# input bar 1

difference = st.number_input("*Average Salary Difference (in $)")
st.caption("*Note: Average Salary Difference is the average increase/decrease of a salary across a player's entire career. \
For example, Derek Jeter's Average Salary Difference between 2012 and 2013 would be \\$1M if his salary was \\$14M in 2012 and \\$15M in 2013")

# input bar 2
age = st.slider('Age', 18, 45, 25)

# input bar 3
wins = st.slider('Wins', 0, 25, 10)

# input bar 4
losses = st.slider('Losses', 0, 25, 10)

# input bar 5
era = st.number_input('ERA')

# input bar 6
games = st.slider('Games Played', 0, 100, 30)

# input bar 7
saves = st.slider('Saves', 0, 50, 0)

# input bar 8
ip = st.slider('Innings Pitched', 0, 350, 150)

# input bar 9
hits = st.slider('Hits Allowed', 0, 300, 150)

# input bar 10
hr = st.slider('Homeruns Allowed', 0, 50, 20)

# input bar 11
so = st.slider('Strikeouts', 0, 350, 150)

# input bar 12
bb = st.slider('Walks', 0, 100, 40)

# if button is pressed
if st.button("Submit"):

    # unpickle the batting model
    pb_model = joblib.load("pb_model.pkl")

    # store inputs into df
    column_names = ['Salary Difference', 'Age', 'W', 'L', 'ERA', 'G', 'SV', 'IP', 'H', 'HR', 'SO', 'BB']
    df = pd.DataFrame([[difference, age, wins, losses, era, games, saves, ip, hits, hr, so, bb]], 
                     columns = column_names)

    # get prediction
    prediction = pb_model.predict(df)

    # convert prediction
    converted = round(np.exp(prediction)[0],0)

    with st.spinner('Calculating...'):
        time.sleep(1)
    st.success('Done!')

    st.dataframe(df)

    # output prediction
    st.header(f"Predicted Player Salary: ${converted:,}")

# header
st.markdown("### How do the predictions compare to 2022 stats thus far?")
st.markdown("###### Updated: Aug 24, 2022")

# 2022 pitching dataframe
pitching_2022_df = pd.read_csv('pitching_merged_2022', index_col = 0)

# reformat 2022 pitching df for model prediction
df_to_predict = pitching_2022_df.drop(columns = ['Name', '2022 Salary'])

# load in model
pb_model = joblib.load("pb_model.pkl")

# make prediction
predictions_2022 = pb_model.predict(df_to_predict)

# Add prediction column
pitching_2022_df["Predicted Salary"] = np.around(np.exp(predictions_2022),0)

# Add value column
pitching_2022_df.loc[pitching_2022_df['Predicted Salary'] > pitching_2022_df['2022 Salary'], 'Value?'] = 'Under-valued'
pitching_2022_df.loc[pitching_2022_df['Predicted Salary'] < pitching_2022_df['2022 Salary'], 'Value?'] = 'Over-valued'

# reorder columns
pitching_2022_df= pitching_2022_df[['Name', '2022 Salary', 'Predicted Salary', 'Value?', 'Avg Career Salary Difference', 'Age', \
                                'W', 'L', 'ERA', 'G', 'SV', 'IP', 'H', 'HR', 'SO', 'BB']]

# formatting as Millions
pitching_2022_df['2022 Salary']  = pitching_2022_df['2022 Salary'] .div(1000000).round(2)
pitching_2022_df['Predicted Salary'] = pitching_2022_df['Predicted Salary'] .div(1000000).round(2)
pitching_2022_df['Avg Career Salary Difference']  = pitching_2022_df['Avg Career Salary Difference'].div(1000000).round(2)

pitching_2022_df = pitching_2022_df.rename(columns = {'2022 Salary':'2022 Salary ($ Millions)',
                                                  'Predicted Salary':'Predicted Salary ($ Millions)',
                                                  'Avg Career Salary Difference':'Avg Career Salary Difference ($ Millions)'})


st.dataframe(pitching_2022_df)
