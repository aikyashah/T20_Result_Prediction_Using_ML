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
df1 = pd.read_csv('deliveries.csv')
df2 = pd.read_csv('workingData.csv')

abandoned = [301,546,571,11340] # Rain abandoned matches with no result
#Remove abandoned matchesmatch_
for a_id in abandoned:
    indexIds1 = df1[df1['match_id'] == a_id].index
    df1.drop(indexIds1, inplace = True)
    indexIds2 = df2[df2['id'] == a_id].index
    df2.drop(indexIds2, inplace = True)

df1['player_dismissed'] = df1['player_dismissed'].fillna(0)

prev_innings  = 1
prev_matchid = 1
teamscore = 0
wickets = 0

for row in df1.itertuples():
    curr_matchid = row.match_id
    curr_innings = row.inning
    if curr_innings > 2:
        continue
    if(curr_matchid != prev_matchid & prev_innings==2):
        df2.loc[df2['id']==prev_matchid,['team2_score']] = teamscore
        teamscore = 0
        wickets = 0
        prev_matchid = curr_matchid
        #curr_innings = row.inning
    elif curr_innings != prev_innings & prev_innings == 1:
        df2.loc[df2['id']==curr_matchid,['target']] = teamscore
        teamscore = 0
        wickets = 0
        prev_innings = 2
    
    if prev_innings==1 & curr_innings == 1:
        teamscore += row.total_runs
        if row.player_dismissed==0:
            wickets = wickets + 0
        else:
            wickets = wickets + 1
        
    if prev_innings == 2 & curr_innings == 2:
        #teamscore += row.total_runs
        #wickets = wickets + 0 if np.isnan(row.player_dismissed) else 1
        
        if row.over == 6 & row.ball == 1:
            df2.loc[df2['id']==curr_matchid,['team2_30_rn']] = teamscore
            df2.loc[df2['id']==curr_matchid,['team2_30_wk']] = wickets
        elif row.over == 11 & row.ball == 1:
            print("The row over is : "+str(row.over))
            df2.loc[df2['id']==curr_matchid,['team2_60_rn']] = teamscore
            df2.loc[df2['id']==curr_matchid,['team2_60_wk']] = wickets
        elif row.over == 16 & row.ball == 1:
            df2.loc[df2['id']==curr_matchid,['team2_90_rn']] = teamscore
            df2.loc[df2['id']==curr_matchid,['team2_90_wk']] = wickets
            
        teamscore += row.total_runs
        if row.player_dismissed==0:
            wickets = wickets + 0
        else:
            wickets = wickets + 1
        
        if wickets==10 & row.over<16:
            df2.loc[df2['id']==curr_matchid,['team2_90_rn']] = teamscore
            df2.loc[df2['id']==curr_matchid,['team2_90_wk']] = wickets

df2['team2_30_ach'] = df2['team2_score'] - df2['team2_30_rn']
df2['team2_60_ach'] = df2['team2_score'] - df2['team2_60_rn']
df2['team2_90_ach'] = df2['team2_score'] - df2['team2_90_rn']

df2['team2_win'] = 1
df2.loc[df2['winner']==df2['team1'],['team2_win']]=0
        
    
    
    