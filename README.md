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

  <h3>1. Statement</h3>
  I aim to employ Machine Learning models to predict the result of T20 cricket matches and find out if applying them during     rain interruptions is statistically feasible, and if so, how does it compare to the DLS method, which is the officialy used method currently.

  <h3>2. Motivation</h3>
  Cricket matches often encounter rain or poor lighting conditions and unfortuately have to stop the play. In this case, the DLS method, which is a statistical formula loosely based on Machine Learning is employed to determine the winner. I wanted to check how accurate DLS method is. I also was curious to check if traditional Machine Learning algorithms, that have so far been used on player level statistics and team level staistics, if employed on match results, can predict the outcomes better than DLS method.
  
  <h3>3. Basic Rules of Cricket</h3>
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

  
  <h3>4. Assumptions and Cleaning:</h3>
  I have made the following adjustments:
  
  * For training ML models, I have considered only those matches that have been completed without rain interruption.
  * My model predicts results, given that Team 1 has completed it's innings.
  * The game can stop at any stage of the Team 2's innings depending on when it rains: After over 5 or over 6 or over 7 and so on ... till over 20. Predicting at each such stage requires at least 15 classifiers for each model. Since this is an introductory project, I have chosen only 3 stages.
  * I have considered 3 stages of stoppage for predictions:
    * Stage 1: When 5 overs are complete (i.e. 25% of the innings of Team 2 is complete)
    * Stage 2: When 10 overs are complete (i.e. 50% of the innings of Team 2 is complete)
    * Stage 3: When 15 overs are complete (i.e. 75% of the innings of Team 2 is complete)
  * If Team 2 wins after super over, it is still considered a win.

  <p> The purpose of the training being so selective is that I want to predict how he game would have developed normally without an exceptional performance. My model does not consider the strength of the oppostion bowlers or the skills of the batsmen still to come or the playing conditions.</p>
  
  <h5>Data Cleaning: </h5>
  File Used: Deliveries.csv, WorkingData.csv
  lines: 179970
  
  Deliveries.csv contains data of ball by ball play of 738 Indian Premier League 20 cricket matches. It does not contain the total score at the end of innings or wickets. WorkingData.csv contains corrosponding match ids and winner data.
  
  I filtered all data of those match Ids that were incomplete or decided by DLS method. It is counter intuitive to compare with DLS method if the training data resluts are decided by the DLS method itself! I filtered data of incomplete matches.
  
  Then I looped through the entire file. If it was first innings, I summed the total runs on each ball and input the final target at the end of 20 oversor 10 wickets, whichever happened first.
  If it was the second innings, I calculated score and wickets after 5 overs, 10 overs and 15 overs.
  
  Finally I input the entire dataframe to a csv file named PreparedData.csv
  
  
  <h3> 5. Methodology and Models</h3>
  For this project I used 3 classifiers for each of the following Machine Learning classification models: Random Forest, SVM, Logistic Regressiona and Naive Bayes. I have considered the following 3 features in the training sets:
  
   * The target score for Team 2 to achieve
   * The number of balls remaining in the innings ( which remains fixed for Stage 1 (i.e. 90), 2 (ie 60) and 3 (ie.e 30)
   * The runs scored at this stage
      
  <h3>6. Model Tuning and Tuning Results:</h3>
  I used grid search method to obtain the best accuracies for eeach of the 4 Machine Learning models. What I observed with respect to SVM was that the kernel function 'linear' gave better results than 'rbf' for stage 1 and stage 2, however, there was not so much of a significant difference between the two kernels. Both gave similar average accuracies. Logistic Regression performed better in training but worse in test. Grid search results did not converge. The accuracies were wildly different in each run. The code for the grid search has been commented out.
  
  <h3> 7. Comparisons of Models and DLS Method:</h3>
  The accuracy of the existing non-ML based system used to predict the winner (ie. the DLS method) was calculated over the entire dataset. This is taken as base reference. The ML models' accuracy is over only 183 records and the DLS is for 731 records.
    <h4> Accuracy at Stage 1:</h4>
    ![Stage1 Accuracy](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/Accuracy%20Charts/Accuracy%20stg1.png)
    <h4> Accuracy at Stage 2:</h4>
    ![Stage2 Accuracy](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/Accuracy%20Charts/Accuracy%20stg2.png)
    <h4> Accuracy at Stage 3: <h4>
  ![Stage3 Accuracy](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/Accuracy%20Charts/Accuracy%20stg3.png)
  
  For stage 3, I compared the accuracies of a model considering only Stage 3 vs the accuracy of that same model considering the previous two stages as well - meaning the feature in the combined stage model will have 12 dimensions as opposed to 4 dimensions in the 'only stage 3' model.
    <h4> Accuracy comparison for Stage 3 Alone vs Stage 3 considering Stage2 and Stage 1:</h4>
    ![Stage3 Accuracy Comparison](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/Accuracy%20Charts/Accuracy%20stg%20comb.png)
    
  <h3>8. ROC Plots:</h3>
  ROC plots indicate the true positive rate of the ML model. The following ROC plots were obtained for stage 1. For more ROC plots look at the directory 'ROC Plots':
  **Stage 1**
    <h4>Random Forest</h4>
    ![Stage 1 ROC RF](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/ROC_Plots/Random%20Forest%20ROC%20Stage%201.png) 
     <h4>Naive Bayes</h4>
     ![Stage 1 ROC NB](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/ROC_Plots/Naive%20Bayes%20ROC%20Stage%201.png)
     <h4>SVM</h4>
     ![Stage 1 ROC NB](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/ROC_Plots/SVM%20ROC%20Stage%201.png) 
     <h4>Logistic Regression</h4>
     ![Stage 1 ROC LR](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/ROC_Plots/Logistic%20Regression%20ROC%20Stage%201.png)
     
  <h3>9. An Attempt at Interpretation with Data and Boundary Plots:</h3>
  
  The intuition behind using these 4 models was the small dimension of the data and the length of the data. As opposed to images and words Neural Networks do not usually work well with simple categorical and spreadsheet type data. Let us look at the 3-D data plots:
  [!Stage1 data]
 (https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/Data%20Plots/3D%20Training%20Data%20Stage%201.png)
  
  [!Stage 2 data](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/Data%20Plots/3D%20Training%20Data%20Stage%202.png)
  
  [!Stage 3 data](https://github.com/aikyashah/T20_Result_Prediction_Using_ML/blob/master/Data%20Plots/3D%20Training%20Data%20Stage%203.png)


  <h3>10. Conclusion:</h3>
  
  As we can see, the statistical method holds strong against the ML models at all the three selected stages.
  
   * Stage 1 with 5 overs completed has low accuracy and hence it is too early to determine winner either by DLS or by ML models.
   * There exists at least 1 ML model that outperforms DLS method at each stage, but no model does it consistently across stages and by a clear margin.
   * More data points and different data organization is needed for better performance.

  <h3>11. Future ideas and expansion:</h3>

  To further my idea, I recommend:
  
  * Gather more data points. Probably a set for 50 over cricket.
  * Randomize balls out of 120 and avoid stages altogether. Calculate scores at random intervals

    

  
  
