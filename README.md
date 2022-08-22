# *Stepping Up To The Plate* - Predicting MLB Player Value & Team Wins with Machine Learning
<p align="center">
  <img 
  src = "https://github.com/eric8395/baseball-analytics/blob/main/images/stars.jpeg"  width="600" height="350" />
<p align="center"> 
Image Source: Forbes
</p> 


Like any other business, a major league baseball team operates very similarly. A franchise has to consider many factors for success including winning and the budgeting for success. Success begins and ends with the construction of a roster and translating a roster of players into wins. But how does a team decide how much they should be paying their roster of players? 

## Business Understanding

Building a competitive team facing a limited budget is a predicament many MLB teams face. There are the big market teams like the New York Yankees, Los Angeles Dodgers, and Boston Redsox that are allowed a larger budget than most other teams. How can the smaller market teams even compete when they can't afford the superstar players to help with winning? This is the foundation of Michael Lewis's <a href="https://en.wikipedia.org/wiki/Moneyball"> Moneyball: The Art of Winning an Unfair Game </a> and the basis for this project. 
![Dash - Wins_Salaries](https://user-images.githubusercontent.com/86889081/185801237-28c22e95-c68d-4296-94a5-930b3282eb0f.png)

The goal is to construct a machine learning model that can predict the salaries of MLB hitters and pitchers using historical baseball data collected from the last 20 years. Additionally, we want to better understand which statistics in baseball contribute the most to predicting overall team wins. 

Using these models, we can better understand the *value* of the baseball player and identify which players are potentially undervalued or overvalued when constructing a roster of players. 

## Data Sources

Baseball player and team data was sourced using the <a href = "https://github.com/jldbc/pybaseball"> PyBaseball</a> package which allows users to scrape Baseball Reference, Baseball Savant, and FanGraphs data. For this analysis, data was collected from Baseball Reference and FanGraphs. 

For a comprehensive list describing all the features/statistics in this notebook; refer to the following documentations links:

- <a href = 'https://library.fangraphs.com/offense/offensive-statistics-list/'> FanGraphs Batting </a> 

- <a href = 'https://library.fangraphs.com/pitching/complete-list-pitching/'> FanGraphs Pitching </a> 

A total of 6 datasets were scraped, cleaned, and formatted for modeling and consist of the following Basic and Advanced tables:

**Batters:**

Basic: 9100 x 28 Features (2000 - 2021)

Advanced: 3200 x 321 Features (2014 - 2021)

**Pitchers:**

Basic: 9400 x 34 Features (2000 - 2021)

Advanced: 3900 x 334 Features (2014 - 2021)

**Teams:**

Basic: 1600 x 61 Features (1960 - 2021)

Advanced: 210 x 634 Features (2014 - 2021)

Basic data consists of traditional baseball statistics that have been collected for decades such as HRs, RBIs, and SOs. Advanced data consist of collected by <a href = "https://www.mlb.com/glossary/statcast"> Statcast</a>, a state-of-the-art tracking technology that allows for the collection and analysis of a massive amount of baseball data, in ways that were never possible in the past. 

Since Statcast is a relatively new data collection technology, we have data isolated between 2014 - 2021. 

## Data Processing & Understanding

The target variable identified for this analysis was player salary for individual batter and pitcher statistics and wins for team statistics. 

**Feature Engineering and Handling**

- **Missing Salaries:** Batter and pitcher data was collected with the pybaseball package and split into basic and advanced stats. Player salaries were collected for each indidvidual year. There were many instances of missing salary values; these values were imputed based on prior salaries available for each player. For example, if Derek Jeter made $15M in 2010, and a missing value existed for 2011, the missing value would be filled with Jeter's previous year salary of $15M. 

- **Missing Advanced Values:** The advanced statcast data consisted of mostly sparse data for very specific data collection columns. For example, the wKN statistic measures a how well a batter/pitcher performed against/using a knuckleball. Knuckleballs are rarely ever thrown in baseball and will therefore have many missing values. In this instance, these missing values would be filled with zeros. 

- **Adjusting for Inflation:** While the dataset was limited to players from the last 21 years, there is variance in the player salaries across the past two decades. To account for this, player salaries were adjusted for inflation using the national CPI index. 

- **Average Salaries:** The batter and pitcher datasets were grouped by each individual player's average salary across all the years that player played. This method effectively removed categorical features such as position and team played since many players played multiple positions and teams across their careers. 

- **Salary Difference Between years:** To further account for variability in player contracts between years, the difference between player salaries each year was added as a feature into the batter and pitcher dataframes. This addresses the large salary difference for players who make a significant amount more in their free agency year.

- **Feature Selection:** To reduce complexity of the modeling, we implemented the use of Sci-Kit Learn's `feature_selection` class and found that the `SelectKBest` method performed the best when it came down to finding the most important features and explaining the variance of the model. Domain knowledge about the game of baseball also came in handy here when selecting features. 

**Preprocessing**

- A 75%-25% train-test split and 5-Fold validation was implemented for assessment on the batting, pitching, and team datasets. 

- A standard scaler was applied to each dataset to prepare for modeling. The target variable of Salary and Wins were log transformed and reverse logged once the data was passed through the modeling process. 

To simplify the process of modeling, a helper function `model_results` was constructed to get individual model results consisting of training, testing, and validation scores. Metrics used for determining model performance are the coefficient of determination (R2), and the root mean square error (RMSE).

**Data Visualization**

When holistically examining the batter and pitcher datasets, it is obvious that the superstars of MLB far outmake the vast majority of the average baseball player. 

The salaries of these superstar players are also outliers and can be further explained by a multitude of factors not captured by the datasets. For example, below is a visualization of some of 2021's highest paid batters in MLB.

<p align="center">
  <img 
  src = "https://github.com/eric8395/baseball-analytics/blob/main/images/Dash%20-%20Batter%20Salaries.png"  width="620" height="600" />
<p align="center"> 

Overall, batters tend to make more on average than the pitcher. Understandably, batters ie. position players, play everyday and will likely be valued at a higher price than the pitcher who may not play every day. 

<p align="center">
  <img 
  src = "https://github.com/eric8395/baseball-analytics/blob/main/images/Dash%20-%20Salaries.png"  width="750" height="450" />
<p align="center"> 



## Modeling


**Grouping Player Salaries**


## Evaluation


**Recommendations & Next Steps**


## Repository Structure

```
├── images_presentation
├── pdfs
├── .gitignore
├── EDA.ipynb
├── Image Classification - Binary.ipynb
└── Image Classification - Multiclass.ipynb
└── README.md
```
