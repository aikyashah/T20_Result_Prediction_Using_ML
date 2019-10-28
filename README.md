# T20_Result_Prediction_Using_ML
Independent Project : Predicting T20 Results where 1st innings has been completed using Machine Learning Models.

Code File: iplCleanerClassifierPredictor.py

This is my first independent project aimed at learning different Machine Learning Models and interpreting data. The topic was selected because I am passionate about cricket. To expand upon my work, please see the ideas at the end of this file.

<h1>Contents:</h1>

1. Statement
1. Motivation
1. Basic Rules of Cricket
1. Assumptions and Cleaning
1. Methodology and Models
1. Model Tuning and Tuning Results
1. Comparision of Models and DLS Method
1. ROC Plots
1. An Attempt at Interpretaion with Data and Boundary plots
1. Conclusion
1. Future ideas and expansion

  1. ###Statement </h3>
  I aim to employ Machine Learning models to predict the result of T20 cricket matches and find out if applying them during     rain interruptions is statistically feasible, and if so, how does it compare to the DLS method, which is the officialy used method currently.

  2. <h3>Motivation</h3>
  Cricket matches often encounter rain or poor lighting conditions and unfortuately have to stop the play. In this case, the DLS method, which is a statistical formula loosely based on Machine Learning is employed to determine the winner. I wanted to check how accurate DLS method is. I also was curious to check if traditional Machine Learning algorithms, that have so far been used on player level statistics and team level staistics, if employed on match results, can predict the outcomes better than DLS method.
  
  3. <h3> Basic Rules of Cricket</h3>
  For those who are not familiar with the rules of cricket, I shall try to explain the rules of play with some Baseball analogy and comparisons. For others, skip to section 4.
    
  * Glossary:
    * Balls = pitches
    * Over = 6 balls
    * Wickets = Outs
    * Each Team bats only once. Each player in a team can bat only once.
  
  * In cricket, there are 3 formats depending on the limit of pitches (called 'balls') in cricket. The longest form goes on for 5 days. My project is based on the shortest format, capped at 120 'balls'(pitches) that usually finishes in 4 hours.
  * Henceforth all rules are explained for this shortest format called the 'T20' cricket (short for Twenty-Twenty cricket).
  * There are only 2 innings, 1 by each team. The decision who sets the target is done by a coin toss.
  * Team 1 bats for 120 balls and tries to score as many runs as possible in 120 balls. 6 balls form 1 over. So 120 balls form 20 such 'overs'. Hence the name T20. 
  * At the end of 20 'overs' or 120 pitches, team 2 takes the chance to bat. It has to score 1 more run than the runs team 1 scored to win the game. It has to achieve this within 20 overs.
  * If Team 2 achieves the target, Team 2 wins. Else it loses. In rare cases it is a tie, which is decided by playing 1 extra over.
  * There are 11 batters (called 'batsmen') in a team. 
  * An innings ends if 10 batsmen get out before 20 overs. An 'out' is called a 'wicket'.
  * Any player **can bat only once**. Hence having less outs is an important resouce. More players a team has still to play, the more risk it can take and hence accelerate the runs scoring.

  
  4. ### Assumptions and Cleaning:
  I have made the following adjustments:
  * For training ML models, I have considered only those matches that have been completed without rain interruption.
  * My model predicts results, given that Team 1 has completed it's innings.
  * The game can stop at any stage of the Team 2's innings depending on when it rains: After over 5 or over 6 or over 7 and so on ... till over 20.
  * I have considered 3 stages of stoppage for predictions: 
    * **Stage 1: When 5 overs are complete** (i.e. 25% of the innings of Team 2 is complete)
    * **Stage 1: When 10 overs are complete** (i.e. 50% of the innings of Team 2 is complete)
    * **Stage 3: When 15 overs are complete** (i.e. 75% of the innings of Team 2 is complete
  
  
