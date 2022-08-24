import numpy as np
import pickle
import pandas as pd
import streamlit as st
import joblib
import time
# from PIL import Image

st.image("https://media.istockphoto.com/photos/rear-view-of-baseball-batter-and-catcher-watching-the-pitch-picture-id1174867119?b=1&k=20&m=1174867119&s=170667a&w=0&h=Lpk2muXoNKWB8dTpak55rqwM1ffEddzgSZsmJeZKEvg=", use_column_width= 'always')

# Title
st.title("Predicting MLB Batter Salaries")

# Subtitle
st.markdown("Enter the following statistics for a batter and \
                    get an estimated salary value.")

# Add sidebar
st.sidebar.markdown("## Predict the Salary of a MLB Hitter!")
st.sidebar.image("https://assets.stickpng.com/images/584d4c0e0a44bd1070d5d495.png", width = 200)
st.sidebar.markdown("How does a team determine how much to pay their players?")

st.sidebar.markdown("#### Built by Eric Au")

# input bar 1
difference = st.number_input("Enter Average Salary Difference $\
    (Average increase/decrease across a player's career so far)")

# input bar 2
age = st.slider('Age', 18, 45, 25)

# input bar 3
hits = st.slider('Hits', 0, 250, 100)

# input bar 4
runs= st.slider('Runs', 0, 200, 50)

# input bar 5
rbi = st.slider('RBIs', 0, 200, 75)

# input bar 6
walks = st.slider('Walks', 0, 250, 50)

# input bar 7
so = st.slider('Strikeouts', 0, 250, 50)

# input bar 8
sb = st.slider('Stolen Bases', 0, 100, 10)

# input bar 9
ops = st.number_input("Enter OPS")

# if button is pressed
if st.button("Submit"):

    # unpickle the batting model
    bb_model = joblib.load("pkl/bb_model.pkl")

    # store inputs into df

    column_names = ['Salary Difference', 'Age', 'H', 'R', 'RBI', 'BB', 'SO', 'SB', 'OPS']
    df = pd.DataFrame([[difference, age, hits, runs, rbi, walks, so, sb, ops]], 
                     columns = column_names)

    # get prediction
    prediction = bb_model.predict(df)

    # convert prediction
    converted = round(np.exp(prediction)[0],0)

    with st.spinner('Calculating...'):
        time.sleep(1)
    st.success('Done!')

    st.dataframe(df)

    # output prediction
    st.subheader(f"Predicted Player Salary: ${converted:,}")

# header
st.markdown("### How do the predictions compare to 2022 stats thus far?")
st.markdown("###### Updated: Aug 24, 2022")

# 2022 batter dataframe
batter_2022_df = pd.read_csv('batting_merged_2022', index_col = 0)
# reformat 2022 batter df for model prediction
df_to_predict = batter_2022_df.drop(columns = ['Name', '2022 Salary'])

# load in model
bb_model = joblib.load("pkl/bb_model.pkl")

# make prediction
predictions_2022 = bb_model.predict(df_to_predict)

# Add prediction column
batter_2022_df["Predicted Salary"] = np.around(np.exp(predictions_2022),0)

# Add value column
batter_2022_df.loc[batter_2022_df['Predicted Salary'] > batter_2022_df['2022 Salary'], 'Value?'] = 'Under-valued'
batter_2022_df.loc[batter_2022_df['Predicted Salary'] < batter_2022_df['2022 Salary'], 'Value?'] = 'Over-valued'

# reorder columns
batter_2022_df = batter_2022_df[['Name', '2022 Salary', 'Predicted Salary', 'Value?', 'Avg Career Salary Difference', 'Age', \
                                'H', 'R', 'RBI', 'BB', 'SO', 'SB', 'OPS']]

st.dataframe(batter_2022_df)