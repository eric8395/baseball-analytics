# *Stepping Up To The Plate* - Predicting MLB Player Value & Team Wins with Machine Learning
<p align="center">
  <img 
  src = "https://github.com/eric8395/baseball-analytics/blob/main/images/stars.jpeg"  width="600" height="350" alt = "test"/>
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

Prior to 


## Modeling


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
