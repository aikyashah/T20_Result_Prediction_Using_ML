#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:08:32 2019

@author: aikya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df1 = pd.read_csv('deliveries.csv')             # The file where we will calculate scores ball by ball
df2 = pd.read_csv('workingData.csv')            # This is the file where we will input our calculated values

abandoned = [301,546,571,11340] # Rain abandoned matches with no result
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
        print("Target set is: ",teamscore)
        target = teamscore+1
        df2.loc[df2['id']==curr_matchid,['target']] = teamscore+1
        teamscore = 0
        wickets = 0
        prev_innings = 2
    
    if prev_innings==1 and curr_innings == 1:
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

# Data Cleaning and Preparing Ends
##############################################################################################


# Start of Using Models

    
    
    