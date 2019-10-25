#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:08:32 2019

@author: aikya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def preprocessFile():
    # Importing the dataset
    df1 = pd.read_csv('deliveries.csv')             # The file where we will calculate scores ball by ball
    df2 = pd.read_csv('workingData.csv')            # This is the file where we will input our calculated values

    abandoned = [301,452, 546,566, 571,11340] # Rain abandoned matches with no result or shortened matches with less tha 11 overs
    #Remove abandoned matches from the dataframe
    for a_id in abandoned:
        indexIds1 = df1[df1['match_id'] == a_id].index
        df1.drop(indexIds1, inplace = True)
        indexIds2 = df2[df2['id'] == a_id].index
        df2.drop(indexIds2, inplace = True)

    #Fill wickets column with 0 to replace Nan if now wickets fell on that ball
    df1['player_dismissed'] = df1['player_dismissed'].fillna(0)

    prev_innings  = 1
    prev_matchid = 1
    teamscore = 0
    wickets = 0

    #Iterate through each ball and calculate the score at regular intervals
    for row in df1.itertuples():
        curr_matchid = row.match_id
        curr_innings = row.inning
        if curr_innings > 2:                                                    # Check for superover condition, and ignore
            continue
        # This condition checks whether we have moved on to the next match in the entries.
        # If so, input our calculated scores at appropraite points in the dataframe2
        if(curr_matchid != prev_matchid and prev_innings==2):
            df2.loc[df2['id']==prev_matchid,['team2_score']] = teamscore
            teamscore = 0
            wickets = 0
            prev_matchid = curr_matchid
            prev_innings = 1
            #curr_innings = row.inning
        elif curr_innings != prev_innings and prev_innings == 1:                # Innings 1 ended, 2nd innings started. Reset values
            #print("Target set is: ",teamscore)
            target = teamscore+1
            df2.loc[df2['id']==curr_matchid,['target']] = teamscore+1
            teamscore = 0
            wickets = 0
            prev_innings = 2

        if prev_innings==1 and curr_innings == 1:                              #  Inside innings 1
            teamscore += row.total_runs
            if row.player_dismissed==0:
                wickets = wickets + 0
            else:
                wickets = wickets + 1

        if prev_innings == 2 and curr_innings == 2:                             # Inside innings 2
            #teamscore += row.total_runs
            #wickets = wickets + 0 if np.isnan(row.player_dismissed) else 1

            #print (row.over, row.ball)
            if int(row.over) == 6 and int(row.ball) == 1:                       # Stage 1: 5 overs complete
                #print ("In loop 1")
                df2.loc[df2['id']==curr_matchid,['team2_30_rn']] = teamscore
                df2.loc[df2['id']==curr_matchid,['team2_30_wk']] = wickets
            elif int(row.over) == 11 and int(row.ball) == 1:                    # Stage 2: 10 overs complete
                #print("In loop 2", row.over, row.ball)
                #print("The row over is : "+str(row.over))
                df2.loc[df2['id']==curr_matchid,['team2_60_rn']] = teamscore
                df2.loc[df2['id']==curr_matchid,['team2_60_wk']] = wickets
            elif int(row.over) == 16 and int(row.ball) == 1:                    # Stage 3: 15 overs complete
                #print("In loop 3")
                df2.loc[df2['id']==curr_matchid,['team2_90_rn']] = teamscore
                df2.loc[df2['id']==curr_matchid,['team2_90_wk']] = wickets

            teamscore += row.total_runs
            if row.player_dismissed==0:
                wickets = wickets + 0
            else:
                wickets = wickets + 1

            if (wickets==10 or teamscore>=target) and row.over<16:              # What if 2nd innings completed before the 10th or 15th over?
                df2.loc[df2['id']==curr_matchid,['team2_90_rn']] = teamscore    # Inpt the final score to Stage 2 and Stage 3 to avoid 'NaN'
                df2.loc[df2['id']==curr_matchid,['team2_90_wk']] = wickets

                if(row.over<11):
                    df2.loc[df2['id'] == curr_matchid, ['team2_60_rn']] = teamscore
                    df2.loc[df2['id'] == curr_matchid, ['team2_60_wk']] = wickets

    df2['team2_30_ach'] = df2['team2_score'] - df2['team2_30_rn']
    df2['team2_60_ach'] = df2['team2_score'] - df2['team2_60_rn']
    df2['team2_90_ach'] = df2['team2_score'] - df2['team2_90_rn']

    # If team 2 wins (chasing team wins), we input 1, else we put 0. This includes result post superover
    df2['team2_win'] = 1
    df2.loc[df2['winner']==df2['team1'],['team2_win']]=0

    dlmethodIds = df2[df2['dl_applied'] == 1].index
    df2.drop(dlmethodIds, inplace=True)

    df2.to_csv(r'PreparedData.csv')                                             # Store Processed Data in a CSV File

# Data Cleaning and Preparing Ends (End of PreprocessFile function)
##############################################################################################
#Column Headers For my reference:
""" 
# 'id', 'season', 'team1', 'team2', 'result', 'target', 'team2_score',     - 7
       'balls_30', 'team2_30_rn', 'team2_30_wk', 'team2_30_ach', 'balls_60',  - 12
       'team2_60_rn', 'team2_60_wk', 'team2_60_ach', 'balls_90', 'team2_90_rn',  -17
       'team2_90_wk', 'team2_90_ach', 'dl_applied', 'winner', 'win_by_runs',     - 22
       'win_by_wickets', 'team2_win'

"""

# Start of Using Models

#preprocessFile()               # Uncomment to process the file again

# Prep
df = pd.read_csv('PreparedData.csv')
X = df.iloc[:,[6,8,9,10,12,13,14,16,17,18]].values
X_30 = df.iloc[:,[6,8,9,10]].values
X_60 = df.iloc[:,[6,12,13,14]].values
X_90 = df.iloc[:,[6,16,17,18]].values
y = df.iloc[:,24].values

# Splitting the dataset into the Training set and Test set  (75%  -  25% split)
from sklearn.model_selection import train_test_split
X_30_train, X_30_test, y_30_train, y_30_test = train_test_split(X_30, y, test_size = 0.25)
X_60_train, X_60_test, y_60_train, y_60_test = train_test_split(X_60, y, test_size = 0.25)
X_90_train, X_90_test, y_90_train, y_90_test = train_test_split(X_90, y, test_size = 0.25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# 10  - cross - validation to check variance
def cross_validation(classifier, X_training_set, y_training_set, stage, model):
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=classifier, X=X_training_set, y=y_training_set, cv=10)
    print("The accuracy of Stage", str(stage) +" for ", model +" is: ",str(accuracies.mean()))
    print("The variance among these is : ",str(accuracies.std()))
    print("--------------------------------------------------------------------------\n")



# 1.  Random Forest

def random_forest():
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier_30 = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=15)
    rf_classifier_30.fit(X_30_train,y_30_train)

    rf_classifier_60 = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=3, min_samples_split=3)
    rf_classifier_60.fit(X_60_train,y_60_train)

    rf_classifier_90 = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=5, min_samples_split=3)
    rf_classifier_90.fit(X_90_train,y_90_train)

    rf_classifier_all = RandomForestClassifier(n_estimators=120, criterion='entropy', max_depth=7, min_samples_split=2)
    rf_classifier_all.fit(X_train,y_train)

    # Predicting the Test set results with Naive Bayes
    rf_y_30_pred = rf_classifier_30.predict(X_30_test)
    rf_y_60_pred = rf_classifier_60.predict(X_60_test)
    rf_y_90_pred = rf_classifier_90.predict(X_90_test)
    rf_y_pred = rf_classifier_all.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    rf_cm_30 = confusion_matrix(y_30_test, rf_y_30_pred)
    rf_cm_60 = confusion_matrix(y_60_test, rf_y_60_pred)
    rf_cm_90 = confusion_matrix(y_90_test, rf_y_90_pred)
    rf_cm = confusion_matrix(y_test, rf_y_pred)

    cross_validation(rf_classifier_30,X_30_train,y_30_train,1, "Random Forest")
    cross_validation(rf_classifier_60, X_60_train, y_60_train, 2, "Random Forest")
    cross_validation(rf_classifier_90, X_90_train, y_90_train, 3, "Random Forest")
    cross_validation(rf_classifier_all, X_train, y_train, 3, "Random Forest")




# Scaling needed for SVM and Naive Bayes
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

X_30_train = sc_X.fit_transform(X_30_train)
X_30_test = sc_X.transform(X_30_test)
#y_30_train = sc_y.fit_transform(y_30_train)

X_60_train = sc_X.fit_transform(X_60_train)
X_60_test = sc_X.transform(X_60_test)
#y_60_train = sc_y.fit_transform(y_60_train)

X_90_train = sc_X.fit_transform(X_90_train)
X_90_test = sc_X.transform(X_90_test)
#y_90_train = sc_y.fit_transform(y_90_train)




# 2. Naive Bayes Prediction


def naive_Bayes():
    from sklearn.naive_bayes import GaussianNB

    nb_classifier_30 = GaussianNB()
    nb_classifier_30.fit(X_30_train, y_30_train)

    nb_classifier_60 = GaussianNB()
    nb_classifier_60.fit(X_60_train, y_60_train)

    nb_classifier_90 = GaussianNB()
    nb_classifier_90.fit(X_90_train, y_90_train)

    nb_classifier_all = GaussianNB()
    nb_classifier_all.fit(X_train, y_train)


    # Predicting the Test set results with Naive Bayes
    nb_y_30_pred = nb_classifier_30.predict(X_30_test)
    nb_y_60_pred = nb_classifier_60.predict(X_60_test)
    nb_y_90_pred = nb_classifier_90.predict(X_90_test)
    nb_y_pred = nb_classifier_all.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    nb_cm_30 = confusion_matrix(y_30_test, nb_y_30_pred)
    nb_cm_60 = confusion_matrix(y_60_test, nb_y_60_pred)
    nb_cm_90 = confusion_matrix(y_90_test, nb_y_90_pred)
    nb_cm = confusion_matrix(y_test, nb_y_pred)

    cross_validation(nb_classifier_30,X_30_train,y_30_train,1, "Naive Bayes")
    cross_validation(nb_classifier_60, X_60_train, y_60_train, 2, "Naive Bayes")
    cross_validation(nb_classifier_90, X_90_train, y_90_train, 3, "Naive Bayes")
    cross_validation(nb_classifier_all, X_train, y_train, 3, "Naive Bayes")



# 3. SVM


def svm():
    from sklearn.svm import SVC

    svc_classifier_30 = SVC(kernel='linear', C=1)
    svc_classifier_30.fit(X_30_train, y_30_train)

    svc_classifier_60 = SVC(kernel='linear', C=1)
    svc_classifier_60.fit(X_60_train, y_60_train)

    svc_classifier_90 = SVC(kernel='rbf', C=2, gamma=0.001)
    svc_classifier_90.fit(X_90_train, y_90_train)

    svc_classifier_all = SVC(kernel='linear', C=1)
    svc_classifier_all.fit(X_train, y_train)

    # Predicting the Test set results with Naive Bayes
    svc_y_30_pred = svc_classifier_30.predict(X_30_test)
    svc_y_60_pred = svc_classifier_60.predict(X_60_test)
    svc_y_90_pred = svc_classifier_90.predict(X_90_test)
    svc_y_pred = svc_classifier_all.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    svc_cm_30 = confusion_matrix(y_30_test, svc_y_30_pred)
    svc_cm_60 = confusion_matrix(y_60_test, svc_y_60_pred)
    svc_cm_90 = confusion_matrix(y_90_test, svc_y_90_pred)
    svc_cm = confusion_matrix(y_test, svc_y_pred)

    # 10 cross-validation to determine variance
    cross_validation(svc_classifier_30,X_30_train,y_30_train,1, "SVM")
    cross_validation(svc_classifier_60, X_60_train, y_60_train, 2, "SVM")
    cross_validation(svc_classifier_90, X_90_train, y_90_train, 3, "SVM")
    cross_validation(svc_classifier_all, X_train, y_train, 3, "SVM")

    from sklearn.model_selection import GridSearchCV
    hyperF = [{'C': [1, 2, 3, 5], 'kernel':['rbf'], 'gamma': [0.001,0.0001, 0.005]}]

    grid_30 = GridSearchCV(estimator=svc_classifier_30, param_grid=hyperF, scoring='accuracy', cv=10, n_jobs=-1)
    grid_30.fit(X_30_train, y_30_train)
    print("\n The best accuracy is given by: ", str(grid_30.best_score_))
    print("\n The best parameters are: ", grid_30.best_params_)

    grid_60 = GridSearchCV(estimator=svc_classifier_60, param_grid=hyperF, scoring='accuracy', cv=10, n_jobs=-1)
    grid_60.fit(X_60_train, y_60_train)
    print("\n The best accuracy is given by: ", str(grid_60.best_score_))
    print("\n The best parameters are: ", grid_60.best_params_)

    grid_90 = GridSearchCV(estimator=svc_classifier_90, param_grid=hyperF, scoring='accuracy', cv=10, n_jobs=-1)
    grid_90.fit(X_90_train, y_90_train)
    print("\n The best accuracy is given by: ", str(grid_90.best_score_))
    print("\n The best parameters are: ", grid_90.best_params_)

    grid_all = GridSearchCV(estimator=svc_classifier_all, param_grid=hyperF, scoring='accuracy', cv=10, n_jobs=-1)
    grid_all.fit(X_train, y_train)
    print("\n The best accuracy is given by: ", str(grid_all.best_score_))
    print("\n The best parameters are: ", grid_all.best_params_)



# 4. Logistic Regression


def logisticRegressionClassifier():
    from sklearn.linear_model import LogisticRegression

    log_classifier_30 = LogisticRegression(solver='liblinear', max_iter=100, penalty='l2')
    log_classifier_30.fit(X_30_train, y_train)

    log_classifier_60 = LogisticRegression(solver='liblinear', max_iter=100, penalty='l2')
    log_classifier_60.fit(X_60_train, y_train)

    log_classifier_90 = LogisticRegression(solver='liblinear', max_iter=100, penalty='l2')
    log_classifier_90.fit(X_90_train, y_train)

    log_classifier_all = LogisticRegression(solver='liblinear', max_iter=100, penalty='l2')
    log_classifier_all.fit(X_train, y_train)

    # Predicting the Test set results with Naive Bayes
    log_y_30_pred = log_classifier_30.predict(X_30_test)
    log_y_60_pred = log_classifier_60.predict(X_60_test)
    log_y_90_pred = log_classifier_90.predict(X_90_test)
    log_y_pred = log_classifier_all.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    log_cm_30 = confusion_matrix(y_30_test, log_y_30_pred)
    log_cm_60 = confusion_matrix(y_60_test, log_y_60_pred)
    log_cm_90 = confusion_matrix(y_90_test, log_y_90_pred)
    log_cm = confusion_matrix(y_test, log_y_pred)

    # 10 cross-validation to determine variance
    cross_validation(log_classifier_30,X_30_train,y_30_train,1, "Logisic Regression")
    cross_validation(log_classifier_60, X_60_train, y_60_train, 2, "Logisic Regression")
    cross_validation(log_classifier_90, X_90_train, y_90_train, 3, "Logisic Regression")
    cross_validation(log_classifier_all, X_train, y_train, 3, "Logisic Regression")
"""
    from sklearn.model_selection import GridSearchCV
    hyperF = [{'C': [0.1, 0.001, 1, 10], 'solver':['liblinear'], 'penalty': ['l1', 'l2'], 'max_iter':[100, 500, 1000, 2000]}]

    grid_30 = GridSearchCV(estimator=log_classifier_30, param_grid=hyperF, scoring='accuracy', cv=10, n_jobs=-1)
    grid_30.fit(X_30_train, y_30_train)
    print("\n The best accuracy is given by: ", str(grid_30.best_score_))
    print("\n The best parameters are: ", grid_30.best_params_)

    grid_60 = GridSearchCV(estimator=log_classifier_60, param_grid=hyperF, scoring='accuracy', cv=10, n_jobs=-1)
    grid_60.fit(X_60_train, y_60_train)
    print("\n The best accuracy is given by: ", str(grid_60.best_score_))
    print("\n The best parameters are: ", grid_60.best_params_)

    grid_90 = GridSearchCV(estimator=log_classifier_90, param_grid=hyperF, scoring='accuracy', cv=10, n_jobs=-1)
    grid_90.fit(X_90_train, y_90_train)
    print("\n The best accuracy is given by: ", str(grid_90.best_score_))
    print("\n The best parameters are: ", grid_90.best_params_)

    grid_all = GridSearchCV(estimator=log_classifier_all, param_grid=hyperF, scoring='accuracy', cv=10, n_jobs=-1)
    grid_all.fit(X_train, y_train)
    print("\n The best accuracy is given by: ", str(grid_all.best_score_))
    print("\n The best parameters are: ", grid_all.best_params_)
"""

def main():
    #preprocessFile()
    random_forest()
    #naive_Bayes()
    #svm()
    #logisticRegressionClassifier()

if __name__ == "__main__":
    main()