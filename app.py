import time
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_bcrypt import Bcrypt
import re
import pandas as pd
import numpy as np
from flask_pymongo import PyMongo
from functools import wraps
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from statistics import stdev
import math
import scipy

IPLdata = pd.read_excel('IPL_Cricket_players.xlsx', header=[0, 1])
IPLRole = pd.read_excel('IPLImage.xlsx')
TESTdata = pd.read_excel('Test_Cricket_players.xlsx', header=[0, 1])
TESTRole = pd.read_excel('TESTImage.xlsx')
ODIdata = pd.read_excel('ODI_Cricket_players.xlsx', header=[0, 1])
ODIRole = pd.read_excel('ODIImage.xlsx')
T20data = pd.read_excel('T20_Cricket_players.xlsx', header=[0, 1])
T20Role = pd.read_excel('T20Image.xlsx')

IPL_players = []
ODI_players = []
TEST_players = []
T20_players = []

def predict_and_result_IPL(combined_players):
    final_data = pd.read_excel("IPL_data_scrapped.xlsx")
    playerData = pd.read_excel("IPLImage.xlsx")
    models = pd.DataFrame()
    latest = pd.DataFrame()
    players_list = combined_players
    for player_name in players_list:
        # print(player_name)
        player_data = final_data[final_data['Player'] == player_name]
        if len(player_data) > 2:
            player_new = player_data.dropna()

            # Predict next runs
            X_runs = player_new[player_new.columns[2:11]]
            y_runs = player_new[player_new.columns[21:22]]
            X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(X_runs, y_runs, random_state=123)
            ridge_runs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_runs = linear_model.Ridge(alpha=j).fit(X_train_runs, y_train_runs)
                ridge_df_runs = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_runs.score(X_train_runs, y_train_runs)),
                     'Test': pd.Series(points_runs.score(X_test_runs, y_test_runs))})
                ridge_runs = ridge_runs._append(ridge_df_runs)
            # print(ridge_runs)
            # Calculate average score
            ridge_runs['Average'] = ridge_runs[['Train', 'Test']].mean(axis=1)

            ridge_runs.sort_values(by='Average', ascending=False, inplace=True)
            k_runs = ridge_runs.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_runs = linear_model.Ridge(alpha=k_runs * 10)
            next_runs.fit(X_train_runs, y_train_runs)
            if len(X_train_runs['Runs Scored']) > 1:
                sd_next_runs = stdev(X_train_runs['Runs Scored'].astype('float'))
            else:
                # Handle empty or single-element case (assign default value, skip, etc.)
                sd_next_runs = 0.0

            # Predict next balls
            X_balls = player_new[player_new.columns[2:11]]
            y_balls = player_new[player_new.columns[22:23]]
            X_train_balls, X_test_balls, y_train_balls, y_test_balls = train_test_split(X_balls, y_balls, test_size=0.2,
                                                                                        random_state=123)
            ridge_balls = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_balls = linear_model.Ridge(alpha=j).fit(X_train_balls, y_train_balls)
                ridge_df_balls = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_balls.score(X_train_balls, y_train_balls)),
                     'Test': pd.Series(points_balls.score(X_test_balls, y_test_balls))})
                ridge_balls = ridge_balls._append(ridge_df_balls)

            # Calculate average score
            ridge_balls['Average'] = ridge_balls[['Train', 'Test']].mean(axis=1)

            ridge_balls.sort_values(by='Average', ascending=False, inplace=True)
            k_balls = ridge_balls.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_balls = linear_model.Ridge(alpha=k_balls * 10)
            next_balls.fit(X_train_balls, y_train_balls)
            if len(X_train_balls['Balls Played']) > 1:
                sd_next_balls = stdev(X_train_balls['Balls Played'].astype('float'))
            else:
                sd_next_balls = 0.0

            # Predict next overs
            X_overs = player_new[player_new.columns[11:21]]
            y_overs = player_new[player_new.columns[23:24]]
            X_train_overs, X_test_overs, y_train_overs, y_test_overs = train_test_split(X_overs, y_overs, test_size=0.2,
                                                                                        random_state=123)
            ridge_overs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_overs = linear_model.Ridge(alpha=j).fit(X_train_overs, y_train_overs)
                ridge_df_overs = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_overs.score(X_train_overs, y_train_overs)),
                     'Test': pd.Series(points_overs.score(X_test_overs, y_test_overs))})
                ridge_overs = ridge_overs._append(ridge_df_overs)

            # Calculate average score
            ridge_overs['Average'] = ridge_overs[['Train', 'Test']].mean(axis=1)

            ridge_overs.sort_values(by='Average', ascending=False, inplace=True)
            k_overs = ridge_overs.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_overs = linear_model.Ridge(alpha=k_overs * 10)
            next_overs.fit(X_train_overs, y_train_overs)
            if len(X_train_overs['Overs Bowled']) > 1:
                sd_next_overs = stdev(X_train_overs['Overs Bowled'].astype('float'))
            else:
                sd_next_overs = 0.0

            # Predict next runs given
            X_runs_given = player_new[player_new.columns[11:21]]
            y_runs_given = player_new[player_new.columns[24:25]]
            X_train_runs_given, X_test_runs_given, y_train_runs_given, y_test_runs_given = train_test_split(
                X_runs_given,
                y_runs_given,
                test_size=0.2,
                random_state=123)
            ridge_runs_given = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_runs_given = linear_model.Ridge(alpha=j).fit(X_train_runs_given, y_train_runs_given)
                ridge_df_runs_given = pd.DataFrame({'Alpha': pd.Series(j), 'Train': pd.Series(
                    points_runs_given.score(X_train_runs_given, y_train_runs_given)), 'Test': pd.Series(
                    points_runs_given.score(X_test_runs_given, y_test_runs_given))})
                ridge_runs_given = ridge_runs_given._append(ridge_df_runs_given)

            # Calculate average score
            ridge_runs_given['Average'] = ridge_runs_given[['Train', 'Test']].mean(axis=1)

            ridge_runs_given.sort_values(by='Average', ascending=False, inplace=True)
            k_runs_given = ridge_runs_given.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_runs_given = linear_model.Ridge(alpha=k_runs_given * 10)
            next_runs_given.fit(X_train_runs_given, y_train_runs_given)
            if len(X_train_runs_given) > 1:
                sd_next_runs_given = stdev(X_train_runs_given['Runs Given'].astype('float'))
            else:
                sd_next_runs_given = 0.0

            X_wkts = player_new[player_new.columns[11:21]]
            y_wkts = player_new[player_new.columns[25:26]]
            X_train_wkts, X_test_wkts, y_train_wkts, y_test_wkts = train_test_split(X_wkts, y_wkts, test_size=0.2,
                                                                                    random_state=123)
            ridge_wkts = pd.DataFrame()
            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_wkts = linear_model.Ridge(alpha=j).fit(X_train_wkts, y_train_wkts)
                ridge_df_wkts = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_wkts.score(X_train_wkts, y_train_wkts)),
                     'Test': pd.Series(points_wkts.score(X_test_wkts, y_test_wkts))})
                ridge_wkts = ridge_wkts._append(ridge_df_wkts)

            # Calculate average score
            ridge_wkts['Average'] = ridge_wkts[['Train', 'Test']].mean(axis=1)
            ridge_wkts.sort_values(by='Average', ascending=False, inplace=True)
            k_wkts = ridge_wkts.head(1)['Alpha'].values[0]
            #
            # Train the model with the best alpha value
            next_wkts = linear_model.Ridge(alpha=k_wkts * 10)
            next_wkts.fit(X_train_wkts, y_train_wkts)
            if len(X_train_wkts) > 1:
                sd_next_wkts = stdev(X_train_wkts['Wickets Taken'].astype('float'))
            else:
                sd_next_wkts = 0.0

            # Get the latest data for the player
            latest = player_data.groupby('Player').tail(1)

            latest.loc[:, 'next_runs'] = next_runs.predict(latest[latest.columns[2:11]])
            latest.loc[:, 'next_balls'] = next_balls.predict(latest[latest.columns[2:11]])
            latest.loc[:, 'next_overs'] = next_overs.predict(latest[latest.columns[11:21]])
            latest.loc[:, 'next_runs_given'] = next_runs_given.predict(latest[latest.columns[11:21]])
            latest.loc[:, 'next_wkts'] = next_wkts.predict(latest[latest.columns[11:21]])

            latest = latest.copy()
            latest.loc[:, 'next_runs_ll_95'], latest.loc[:, 'next_runs_ul_95'] = latest[
                                                                                     'next_runs'] - scipy.stats.norm.ppf(
                .95) * (sd_next_runs / math.sqrt(len(X_train_runs))), latest['next_runs'] + scipy.stats.norm.ppf(
                .95) * (sd_next_runs / math.sqrt(len(X_train_runs)))
            latest.loc[:, 'next_balls_ll_95'], latest.loc[:, 'next_balls_ul_95'] = latest[
                                                                                       'next_balls'] - scipy.stats.norm.ppf(
                .95) * (sd_next_balls / math.sqrt(len(X_train_balls))), latest['next_balls'] + scipy.stats.norm.ppf(
                .95) * (sd_next_balls / math.sqrt(len(X_train_balls)))
            latest.loc[:, 'next_overs_ll_95'], latest.loc[:, 'next_overs_ul_95'] = latest[
                                                                                       'next_overs'] - scipy.stats.norm.ppf(
                .95) * (sd_next_overs / math.sqrt(len(X_train_overs))), latest['next_overs'] + scipy.stats.norm.ppf(
                .95) * (sd_next_overs / math.sqrt(len(X_train_overs)))
            latest.loc[:, 'next_runs_given_ll_95'], latest.loc[:, 'next_runs_given_ul_95'] = latest[
                                                                                                 'next_runs_given'] - scipy.stats.norm.ppf(
                .95) * (sd_next_runs_given / math.sqrt(len(X_train_runs_given))), latest[
                                                                                                 'next_runs_given'] + scipy.stats.norm.ppf(
                .95) * (sd_next_runs_given / math.sqrt(len(X_train_runs_given)))
            latest.loc[:, 'next_wkts_ll_95'], latest.loc[:, 'next_wkts_ul_95'] = latest[
                                                                                     'next_wkts'] - scipy.stats.norm.ppf(
                .95) * (sd_next_wkts / math.sqrt(len(X_train_wkts))), latest['next_wkts'] + scipy.stats.norm.ppf(
                .95) * (sd_next_wkts / math.sqrt(len(X_train_wkts)))
            print(player_name)
            models = models._append(latest)
            # print(models.columns)
            # print(player_name, " is added")

    models['next_runs_given'] = np.where(models['next_overs'] > 4, models['next_runs_given'] / models['next_overs'] * 4,
                                         models['next_runs_given'])
    models['next_runs_given_ll_95'] = np.where(models['next_overs'] > 4,
                                               models['next_runs_given_ll_95'] / models['next_overs'] * 4,
                                               models['next_runs_given_ll_95'])
    models['next_runs_given_ul_95'] = np.where(models['next_overs'] > 4,
                                               models['next_runs_given_ul_95'] / models['next_overs'] * 4,
                                               models['next_runs_given_ul_95'])

    # Limiting next_overs to a maximum of 4
    models['next_overs'] = np.where(models['next_overs'] > 4, 4, models['next_overs'])
    models['next_overs_ll_95'] = np.where(models['next_overs_ll_95'] > 4, 4, models['next_overs_ll_95'])
    models['next_overs_ul_95'] = np.where(models['next_overs_ul_95'] > 4, 4, models['next_overs_ul_95'])

    # Adjusting next_runs based on next_balls
    models['next_runs'] = np.where(models['next_balls'] < 0, 10, models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_balls'] < 0, 12, models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_balls'] < 0, 14, models['next_runs_ul_95'])

    # Setting next_runs to a minimum of 1
    models['next_runs'] = np.where(models['next_runs'] < 0, 11, models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_runs_ll_95'] < 0, 12, models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_runs_ul_95'] < 0, 13, models['next_runs_ul_95'])

    # Adjusting next_runs based on next_balls if next_balls > 100
    models['next_runs'] = np.where(models['next_balls'] > 24, models['next_runs'] / models['next_balls'] * 24,
                                   models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_balls'] > 24,
                                         models['next_runs_ll_95'] / models['next_balls'] * 24,
                                         models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_balls'] > 24,
                                         models['next_runs_ul_95'] / models['next_balls'] * 24,
                                         models['next_runs_ul_95'])

    # Limiting next_balls to a maximum of 5
    models['next_balls'] = np.where(models['next_balls'] > 24, 24, models['next_balls'])
    models['next_balls_ll_95'] = np.where(models['next_balls_ll_95'] > 24, 24, models['next_balls_ll_95'])
    models['next_balls_ul_95'] = np.where(models['next_balls_ul_95'] > 24, 24, models['next_balls_ul_95'])

    # Setting next_balls to a minimum of 1
    models['next_balls'] = np.where(models['next_balls'] < 0, 1, models['next_balls'])
    models['next_balls_ll_95'] = np.where(models['next_balls_ll_95'] < 0, 1, models['next_balls_ll_95'])
    models['next_balls_ul_95'] = np.where(models['next_balls_ul_95'] < 0, 1, models['next_balls_ul_95'])

    # Setting next_wkts to a minimum of 1
    models['next_wkts'] = np.where(models['next_wkts'] < 0, 0, models['next_wkts'])
    models['next_wkts_ll_95'] = np.where(models['next_wkts_ll_95'] < 0, 0, models['next_wkts_ll_95'])
    models['next_wkts_ul_95'] = np.where(models['next_wkts_ul_95'] < 0, 0, models['next_wkts_ul_95'])


    models['next_runs'] = round(models['next_runs'], 0)
    models['next_runs_ll_95'] = round(models['next_runs_ll_95'], 0)
    models['next_runs_ul_95'] = round(models['next_runs_ul_95'], 0)

    models['next_balls'] = round(models['next_balls'], 0)
    models['next_balls_ll_95'] = round(models['next_balls_ll_95'], 0)
    models['next_balls_ul_95'] = round(models['next_balls_ul_95'], 0)

    models['next_wkts'] = round(models['next_wkts'], 0)
    models['next_wkts_ll_95'] = round(models['next_wkts_ll_95'], 0)
    models['next_wkts_ul_95'] = round(models['next_wkts_ul_95'], 0)

    models['next_runs_given'] = round(models['next_runs_given'], 0)
    models['next_runs_given_ll_95'] = round(models['next_runs_given_ll_95'], 0)
    models['next_runs_given_ul_95'] = round(models['next_runs_given_ul_95'], 0)

    models['next_overs'] = round(models['next_overs'], 0)
    models['next_overs_ll_95'] = round(models['next_overs_ll_95'], 0)
    models['next_overs_ul_95'] = round(models['next_overs_ul_95'], 0)
    models.to_excel('modelIPL.xlsx')
    merged_df = pd.merge(models, playerData, on='Player', how='left')
    merged_df.to_excel('mergedIPL.xlsx')
    # print('successfully done')
def playing_eleven_IPL():
    merged_df = pd.read_excel('mergedIPL.xlsx')
    batsman_data = merged_df[merged_df['Role'] == 'Batter']
    bowler_data = merged_df[merged_df['Role'] == 'Bowler']
    wicketK_data = merged_df[merged_df['Role'] == 'WK Keeper - Batter']
    allrounder_data = merged_df[merged_df['Role'] == 'All-Rounder']

    batsman = batsman_data.sort_values(by='next_runs', ascending=False).iloc[:5]
    bowler = bowler_data.sort_values(by='next_wkts', ascending=False).iloc[:3]
    allrounder = allrounder_data.sort_values(by=['next_wkts', 'next_runs'], ascending=[False, False]).iloc[:2]
    if allrounder['next_runs'].iloc[0] == 0:
        allrounder = allrounder_data.sort_values(by=['next_wkts', 'next_runs'], ascending=[False, False]).iloc[1:3]
    wicketK = wicketK_data.sort_values(by='next_runs', ascending=False).iloc[:1]

    playing_eleven = pd.concat([batsman, allrounder, wicketK, bowler])
    return playing_eleven
def predict_and_result_ODI(combined_players):
    final_data = pd.read_excel("ODI_data_scrapped.xlsx")
    playerData = pd.read_excel("ODIImage.xlsx")
    models = pd.DataFrame()
    latest = pd.DataFrame()
    players_list = combined_players
    for player_name in players_list:

        player_data = final_data[final_data['Player'] == player_name]
        print(player_name)
        if len(player_data) > 2:
            player_new = player_data.dropna()
            X_runs = player_new[player_new.columns[2:11]]
            y_runs = player_new[player_new.columns[21:22]]
            X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(X_runs, y_runs, random_state=123)
            ridge_runs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_runs = linear_model.Ridge(alpha=j).fit(X_train_runs, y_train_runs)
                ridge_df_runs = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_runs.score(X_train_runs, y_train_runs)),
                     'Test': pd.Series(points_runs.score(X_test_runs, y_test_runs))})
                ridge_runs = ridge_runs._append(ridge_df_runs)
            # print(ridge_runs)
            # Calculate average score
            ridge_runs['Average'] = ridge_runs[['Train', 'Test']].mean(axis=1)

            ridge_runs.sort_values(by='Average', ascending=False, inplace=True)
            k_runs = ridge_runs.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_runs = linear_model.Ridge(alpha=k_runs * 10)
            next_runs.fit(X_train_runs, y_train_runs)
            if len(X_train_runs['Runs Scored']) > 1:
                sd_next_runs = stdev(X_train_runs['Runs Scored'].astype('float'))
            else:
                # Handle empty or single-element case (assign default value, skip, etc.)
                sd_next_runs = 0.0

            # Predict next balls
            X_balls = player_new[player_new.columns[2:11]]
            y_balls = player_new[player_new.columns[22:23]]
            X_train_balls, X_test_balls, y_train_balls, y_test_balls = train_test_split(X_balls, y_balls, test_size=0.2,
                                                                                        random_state=123)
            ridge_balls = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_balls = linear_model.Ridge(alpha=j).fit(X_train_balls, y_train_balls)
                ridge_df_balls = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_balls.score(X_train_balls, y_train_balls)),
                     'Test': pd.Series(points_balls.score(X_test_balls, y_test_balls))})
                ridge_balls = ridge_balls._append(ridge_df_balls)

            # Calculate average score
            ridge_balls['Average'] = ridge_balls[['Train', 'Test']].mean(axis=1)

            ridge_balls.sort_values(by='Average', ascending=False, inplace=True)
            k_balls = ridge_balls.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_balls = linear_model.Ridge(alpha=k_balls * 10)
            next_balls.fit(X_train_balls, y_train_balls)
            if len(X_train_balls['Balls Played']) > 1:
                sd_next_balls = stdev(X_train_balls['Balls Played'].astype('float'))
            else:
                sd_next_balls = 0.0

            # Predict next overs
            X_overs = player_new[player_new.columns[11:21]]
            y_overs = player_new[player_new.columns[23:24]]
            X_train_overs, X_test_overs, y_train_overs, y_test_overs = train_test_split(X_overs, y_overs, test_size=0.2,
                                                                                        random_state=123)
            ridge_overs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_overs = linear_model.Ridge(alpha=j).fit(X_train_overs, y_train_overs)
                ridge_df_overs = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_overs.score(X_train_overs, y_train_overs)),
                     'Test': pd.Series(points_overs.score(X_test_overs, y_test_overs))})
                ridge_overs = ridge_overs._append(ridge_df_overs)

            # Calculate average score
            ridge_overs['Average'] = ridge_overs[['Train', 'Test']].mean(axis=1)

            ridge_overs.sort_values(by='Average', ascending=False, inplace=True)
            k_overs = ridge_overs.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_overs = linear_model.Ridge(alpha=k_overs * 10)
            next_overs.fit(X_train_overs, y_train_overs)
            if len(X_train_overs['Overs Bowled']) > 1:
                sd_next_overs = stdev(X_train_overs['Overs Bowled'].astype('float'))
            else:
                sd_next_overs = 0.0

            # Predict next runs given
            X_runs_given = player_new[player_new.columns[11:21]]
            y_runs_given = player_new[player_new.columns[24:25]]
            X_train_runs_given, X_test_runs_given, y_train_runs_given, y_test_runs_given = train_test_split(
                X_runs_given,
                y_runs_given,
                test_size=0.2,
                random_state=123)
            ridge_runs_given = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_runs_given = linear_model.Ridge(alpha=j).fit(X_train_runs_given, y_train_runs_given)
                ridge_df_runs_given = pd.DataFrame({'Alpha': pd.Series(j), 'Train': pd.Series(
                    points_runs_given.score(X_train_runs_given, y_train_runs_given)), 'Test': pd.Series(
                    points_runs_given.score(X_test_runs_given, y_test_runs_given))})
                ridge_runs_given = ridge_runs_given._append(ridge_df_runs_given)

            # Calculate average score
            ridge_runs_given['Average'] = ridge_runs_given[['Train', 'Test']].mean(axis=1)

            ridge_runs_given.sort_values(by='Average', ascending=False, inplace=True)
            k_runs_given = ridge_runs_given.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_runs_given = linear_model.Ridge(alpha=k_runs_given * 10)
            next_runs_given.fit(X_train_runs_given, y_train_runs_given)
            if len(X_train_runs_given) > 1:
                sd_next_runs_given = stdev(X_train_runs_given['Runs Given'].astype('float'))
            else:
                sd_next_runs_given = 0.0

            X_wkts = player_new[player_new.columns[11:21]]
            y_wkts = player_new[player_new.columns[25:26]]
            X_train_wkts, X_test_wkts, y_train_wkts, y_test_wkts = train_test_split(X_wkts, y_wkts, test_size=0.2,
                                                                                    random_state=123)
            ridge_wkts = pd.DataFrame()
            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_wkts = linear_model.Ridge(alpha=j).fit(X_train_wkts, y_train_wkts)
                ridge_df_wkts = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_wkts.score(X_train_wkts, y_train_wkts)),
                     'Test': pd.Series(points_wkts.score(X_test_wkts, y_test_wkts))})
                ridge_wkts = ridge_wkts._append(ridge_df_wkts)

            # Calculate average score
            ridge_wkts['Average'] = ridge_wkts[['Train', 'Test']].mean(axis=1)
            ridge_wkts.sort_values(by='Average', ascending=False, inplace=True)
            k_wkts = ridge_wkts.head(1)['Alpha'].values[0]
            #
            # Train the model with the best alpha value
            next_wkts = linear_model.Ridge(alpha=k_wkts * 10)
            next_wkts.fit(X_train_wkts, y_train_wkts)
            if len(X_train_wkts) > 1:
                sd_next_wkts = stdev(X_train_wkts['Wickets Taken'].astype('float'))
            else:
                sd_next_wkts = 0.0

            # Get the latest data for the player
            latest = player_data.groupby('Player').tail(1)

            latest.loc[:, 'next_runs'] = next_runs.predict(latest[latest.columns[2:11]])
            latest.loc[:, 'next_balls'] = next_balls.predict(latest[latest.columns[2:11]])
            latest.loc[:, 'next_overs'] = next_overs.predict(latest[latest.columns[11:21]])
            latest.loc[:, 'next_runs_given'] = next_runs_given.predict(latest[latest.columns[11:21]])
            latest.loc[:, 'next_wkts'] = next_wkts.predict(latest[latest.columns[11:21]])

            latest = latest.copy()
            latest.loc[:, 'next_runs_ll_95'], latest.loc[:, 'next_runs_ul_95'] = latest[
                                                                                     'next_runs'] - scipy.stats.norm.ppf(
                .95) * (sd_next_runs / math.sqrt(len(X_train_runs))), latest['next_runs'] + scipy.stats.norm.ppf(
                .95) * (sd_next_runs / math.sqrt(len(X_train_runs)))
            latest.loc[:, 'next_balls_ll_95'], latest.loc[:, 'next_balls_ul_95'] = latest[
                                                                                       'next_balls'] - scipy.stats.norm.ppf(
                .95) * (sd_next_balls / math.sqrt(len(X_train_balls))), latest['next_balls'] + scipy.stats.norm.ppf(
                .95) * (sd_next_balls / math.sqrt(len(X_train_balls)))
            latest.loc[:, 'next_overs_ll_95'], latest.loc[:, 'next_overs_ul_95'] = latest[
                                                                                       'next_overs'] - scipy.stats.norm.ppf(
                .95) * (sd_next_overs / math.sqrt(len(X_train_overs))), latest['next_overs'] + scipy.stats.norm.ppf(
                .95) * (sd_next_overs / math.sqrt(len(X_train_overs)))
            latest.loc[:, 'next_runs_given_ll_95'], latest.loc[:, 'next_runs_given_ul_95'] = latest[
                                                                                                 'next_runs_given'] - scipy.stats.norm.ppf(
                .95) * (sd_next_runs_given / math.sqrt(len(X_train_runs_given))), latest[
                                                                                                 'next_runs_given'] + scipy.stats.norm.ppf(
                .95) * (sd_next_runs_given / math.sqrt(len(X_train_runs_given)))
            latest.loc[:, 'next_wkts_ll_95'], latest.loc[:, 'next_wkts_ul_95'] = latest[
                                                                                     'next_wkts'] - scipy.stats.norm.ppf(
                .95) * (sd_next_wkts / math.sqrt(len(X_train_wkts))), latest['next_wkts'] + scipy.stats.norm.ppf(
                .95) * (sd_next_wkts / math.sqrt(len(X_train_wkts)))
            models = models._append(latest)

    models['next_runs_given'] = np.where(models['next_overs'] > 10, models['next_runs_given'] / models['next_overs'] * 10,
                                         models['next_runs_given'])
    models['next_runs_given_ll_95'] = np.where(models['next_overs'] > 10,
                                               models['next_runs_given_ll_95'] / models['next_overs'] * 10,
                                               models['next_runs_given_ll_95'])
    models['next_runs_given_ul_95'] = np.where(models['next_overs'] > 10,
                                               models['next_runs_given_ul_95'] / models['next_overs'] * 10,
                                               models['next_runs_given_ul_95'])

    # Limiting next_overs to a maximum of 4
    models['next_overs'] = np.where(models['next_overs'] > 10, 10, models['next_overs'])
    models['next_overs_ll_95'] = np.where(models['next_overs_ll_95'] > 10, 10, models['next_overs_ll_95'])
    models['next_overs_ul_95'] = np.where(models['next_overs_ul_95'] > 10, 10, models['next_overs_ul_95'])

    # Adjusting next_runs based on next_balls
    models['next_runs'] = np.where(models['next_balls'] < 0, 0, models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_balls'] < 0, 0, models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_balls'] < 0, 0, models['next_runs_ul_95'])

    # Setting next_runs to a minimum of 1
    models['next_runs'] = np.where(models['next_runs'] < 0, 1, models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_runs_ll_95'] < 0, 1, models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_runs_ul_95'] < 0, 1, models['next_runs_ul_95'])

    # Adjusting next_runs based on next_balls if next_balls > 100
    models['next_runs'] = np.where(models['next_balls'] > 60, models['next_runs'] / models['next_balls'] * 60,
                                   models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_balls'] > 60,
                                         models['next_runs_ll_95'] / models['next_balls'] * 60,
                                         models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_balls'] > 100,
                                         models['next_runs_ul_95'] / models['next_balls'] * 60,
                                         models['next_runs_ul_95'])

    # Limiting next_balls to a maximum of 5
    models['next_balls'] = np.where(models['next_balls'] > 60, 60, models['next_balls'])
    models['next_balls_ll_95'] = np.where(models['next_balls_ll_95'] > 60, 60, models['next_balls_ll_95'])
    models['next_balls_ul_95'] = np.where(models['next_balls_ul_95'] > 60, 60, models['next_balls_ul_95'])

    # Setting next_balls to a minimum of 1
    models['next_balls'] = np.where(models['next_balls'] < 0, 1, models['next_balls'])
    models['next_balls_ll_95'] = np.where(models['next_balls_ll_95'] < 0, 1, models['next_balls_ll_95'])
    models['next_balls_ul_95'] = np.where(models['next_balls_ul_95'] < 0, 1, models['next_balls_ul_95'])

    # Setting next_wkts to a minimum of 1
    models['next_wkts'] = np.where(models['next_wkts'] < 0, 0, models['next_wkts'])
    models['next_wkts_ll_95'] = np.where(models['next_wkts_ll_95'] < 0, 0, models['next_wkts_ll_95'])
    models['next_wkts_ul_95'] = np.where(models['next_wkts_ul_95'] < 0, 0, models['next_wkts_ul_95'])

    # Rounding values to 0 decimal places
    models['next_runs'] = round(models['next_runs'], 0)
    models['next_runs_ll_95'] = round(models['next_runs_ll_95'], 0)
    models['next_runs_ul_95'] = round(models['next_runs_ul_95'], 0)
    models['next_balls'] = round(models['next_balls'], 0)
    models['next_balls_ll_95'] = round(models['next_balls_ll_95'], 0)
    models['next_balls_ul_95'] = round(models['next_balls_ul_95'], 0)

    models['next_wkts'] = round(models['next_wkts'], 0)
    models['next_wkts_ll_95'] = round(models['next_wkts_ll_95'], 0)
    models['next_wkts_ul_95'] = round(models['next_wkts_ul_95'], 0)

    models['next_runs_given'] = round(models['next_runs_given'], 0)
    models['next_runs_given_ll_95'] = round(models['next_runs_given_ll_95'], 0)
    models['next_runs_given_ul_95'] = round(models['next_runs_given_ul_95'], 0)

    models['next_overs'] = round(models['next_overs'], 0)
    models['next_overs_ll_95'] = round(models['next_overs_ll_95'], 0)
    models['next_overs_ul_95'] = round(models['next_overs_ul_95'], 0)
    models.to_excel('modelODI.xlsx')
    merged_df = pd.merge(models, playerData, on='Player', how='left')
    merged_df.to_excel('mergedODI.xlsx')
def playing_eleven_ODI():
    merged_df = pd.read_excel('mergedODI.xlsx')
    batsman_data = merged_df[merged_df['Role'] == 'Batter']
    bowler_data = merged_df[merged_df['Role'] == 'Bowler']
    wicketK_data = merged_df[merged_df['Role'] == 'WK Keeper - Batter']
    allrounder_data = merged_df[merged_df['Role'] == 'Allrounder']

    batsman = batsman_data.sort_values(by='next_runs', ascending=False).iloc[:5]
    bowler = bowler_data.sort_values(by='next_wkts', ascending=False).iloc[:3]
    allrounder = allrounder_data.sort_values(by=['next_wkts', 'next_runs'], ascending=[False, False]).iloc[:2]
    # print(allrounder.columns)
    if allrounder['next_runs'].iloc[0] == 0:
        allrounder = allrounder_data.sort_values(by=['next_wkts', 'next_runs'], ascending=[False, False]).iloc[1:3]

    wicketK = wicketK_data.sort_values(by='next_runs', ascending=False).iloc[:1]
    playing_eleven = pd.concat([batsman, allrounder, wicketK, bowler])
    return playing_eleven
def predict_and_result_TEST(combined_players):
    final_data = pd.read_excel("Test_data_scrapped.xlsx")
    playerData = pd.read_excel("TESTImage.xlsx")
    models = pd.DataFrame()
    latest = pd.DataFrame()
    players_list = combined_players
    for player_name in players_list:
        # print(player_name)
        player_data = final_data[final_data['Player'] == player_name]
        if len(player_data) > 2:
            player_new = player_data.dropna()

            # Predict next runs
            X_runs = player_new[player_new.columns[2:11]]
            y_runs = player_new[player_new.columns[21:22]]
            X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(X_runs, y_runs, random_state=123)
            ridge_runs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_runs = linear_model.Ridge(alpha=j).fit(X_train_runs, y_train_runs)
                ridge_df_runs = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_runs.score(X_train_runs, y_train_runs)),
                     'Test': pd.Series(points_runs.score(X_test_runs, y_test_runs))})
                ridge_runs = ridge_runs._append(ridge_df_runs)
            # print(ridge_runs)
            # Calculate average score
            ridge_runs['Average'] = ridge_runs[['Train', 'Test']].mean(axis=1)

            ridge_runs.sort_values(by='Average', ascending=False, inplace=True)
            k_runs = ridge_runs.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_runs = linear_model.Ridge(alpha=k_runs * 10)
            next_runs.fit(X_train_runs, y_train_runs)
            if len(X_train_runs['Runs Scored']) > 1:
                sd_next_runs = stdev(X_train_runs['Runs Scored'].astype('float'))
            else:
                # Handle empty or single-element case (assign default value, skip, etc.)
                sd_next_runs = 0.0

            # Predict next balls
            X_balls = player_new[player_new.columns[2:11]]
            y_balls = player_new[player_new.columns[22:23]]
            X_train_balls, X_test_balls, y_train_balls, y_test_balls = train_test_split(X_balls, y_balls, test_size=0.2,
                                                                                        random_state=123)
            ridge_balls = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_balls = linear_model.Ridge(alpha=j).fit(X_train_balls, y_train_balls)
                ridge_df_balls = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_balls.score(X_train_balls, y_train_balls)),
                     'Test': pd.Series(points_balls.score(X_test_balls, y_test_balls))})
                ridge_balls = ridge_balls._append(ridge_df_balls)

            # Calculate average score
            ridge_balls['Average'] = ridge_balls[['Train', 'Test']].mean(axis=1)

            ridge_balls.sort_values(by='Average', ascending=False, inplace=True)
            k_balls = ridge_balls.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_balls = linear_model.Ridge(alpha=k_balls * 10)
            next_balls.fit(X_train_balls, y_train_balls)
            if len(X_train_balls['Balls Played']) > 1:
                sd_next_balls = stdev(X_train_balls['Balls Played'].astype('float'))
            else:
                sd_next_balls = 0.0

            # Predict next overs
            X_overs = player_new[player_new.columns[11:21]]
            y_overs = player_new[player_new.columns[23:24]]
            X_train_overs, X_test_overs, y_train_overs, y_test_overs = train_test_split(X_overs, y_overs, test_size=0.2,
                                                                                        random_state=123)
            ridge_overs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_overs = linear_model.Ridge(alpha=j).fit(X_train_overs, y_train_overs)
                ridge_df_overs = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_overs.score(X_train_overs, y_train_overs)),
                     'Test': pd.Series(points_overs.score(X_test_overs, y_test_overs))})
                ridge_overs = ridge_overs._append(ridge_df_overs)

            # Calculate average score
            ridge_overs['Average'] = ridge_overs[['Train', 'Test']].mean(axis=1)

            ridge_overs.sort_values(by='Average', ascending=False, inplace=True)
            k_overs = ridge_overs.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_overs = linear_model.Ridge(alpha=k_overs * 10)
            next_overs.fit(X_train_overs, y_train_overs)
            if len(X_train_overs['Overs Bowled']) > 1:
                sd_next_overs = stdev(X_train_overs['Overs Bowled'].astype('float'))
            else:
                sd_next_overs = 0.0

            # Predict next runs given
            X_runs_given = player_new[player_new.columns[11:21]]
            y_runs_given = player_new[player_new.columns[24:25]]
            X_train_runs_given, X_test_runs_given, y_train_runs_given, y_test_runs_given = train_test_split(
                X_runs_given,
                y_runs_given,
                test_size=0.2,
                random_state=123)
            ridge_runs_given = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_runs_given = linear_model.Ridge(alpha=j).fit(X_train_runs_given, y_train_runs_given)
                ridge_df_runs_given = pd.DataFrame({'Alpha': pd.Series(j), 'Train': pd.Series(
                    points_runs_given.score(X_train_runs_given, y_train_runs_given)), 'Test': pd.Series(
                    points_runs_given.score(X_test_runs_given, y_test_runs_given))})
                ridge_runs_given = ridge_runs_given._append(ridge_df_runs_given)

            # Calculate average score
            ridge_runs_given['Average'] = ridge_runs_given[['Train', 'Test']].mean(axis=1)

            ridge_runs_given.sort_values(by='Average', ascending=False, inplace=True)
            k_runs_given = ridge_runs_given.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_runs_given = linear_model.Ridge(alpha=k_runs_given * 10)
            next_runs_given.fit(X_train_runs_given, y_train_runs_given)
            if len(X_train_runs_given) > 1:
                sd_next_runs_given = stdev(X_train_runs_given['Runs Given'].astype('float'))
            else:
                sd_next_runs_given = 0.0

            X_wkts = player_new[player_new.columns[11:21]]
            y_wkts = player_new[player_new.columns[25:26]]
            X_train_wkts, X_test_wkts, y_train_wkts, y_test_wkts = train_test_split(X_wkts, y_wkts, test_size=0.2,
                                                                                    random_state=123)
            ridge_wkts = pd.DataFrame()
            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_wkts = linear_model.Ridge(alpha=j).fit(X_train_wkts, y_train_wkts)
                ridge_df_wkts = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_wkts.score(X_train_wkts, y_train_wkts)),
                     'Test': pd.Series(points_wkts.score(X_test_wkts, y_test_wkts))})
                ridge_wkts = ridge_wkts._append(ridge_df_wkts)

            # Calculate average score
            ridge_wkts['Average'] = ridge_wkts[['Train', 'Test']].mean(axis=1)
            ridge_wkts.sort_values(by='Average', ascending=False, inplace=True)
            k_wkts = ridge_wkts.head(1)['Alpha'].values[0]
            #
            # Train the model with the best alpha value
            next_wkts = linear_model.Ridge(alpha=k_wkts * 10)
            next_wkts.fit(X_train_wkts, y_train_wkts)
            if len(X_train_wkts) > 1:
                sd_next_wkts = stdev(X_train_wkts['Wickets Taken'].astype('float'))
            else:
                sd_next_wkts = 0.0

            # Get the latest data for the player
            latest = player_data.groupby('Player').tail(1)

            latest.loc[:, 'next_runs'] = next_runs.predict(latest[latest.columns[2:11]])
            latest.loc[:, 'next_balls'] = next_balls.predict(latest[latest.columns[2:11]])
            latest.loc[:, 'next_overs'] = next_overs.predict(latest[latest.columns[11:21]])
            latest.loc[:, 'next_runs_given'] = next_runs_given.predict(latest[latest.columns[11:21]])
            latest.loc[:, 'next_wkts'] = next_wkts.predict(latest[latest.columns[11:21]])

            latest = latest.copy()
            latest.loc[:, 'next_runs_ll_95'], latest.loc[:, 'next_runs_ul_95'] = latest[
                                                                                     'next_runs'] - scipy.stats.norm.ppf(
                .95) * (sd_next_runs / math.sqrt(len(X_train_runs))), latest['next_runs'] + scipy.stats.norm.ppf(
                .95) * (sd_next_runs / math.sqrt(len(X_train_runs)))
            latest.loc[:, 'next_balls_ll_95'], latest.loc[:, 'next_balls_ul_95'] = latest[
                                                                                       'next_balls'] - scipy.stats.norm.ppf(
                .95) * (sd_next_balls / math.sqrt(len(X_train_balls))), latest['next_balls'] + scipy.stats.norm.ppf(
                .95) * (sd_next_balls / math.sqrt(len(X_train_balls)))
            latest.loc[:, 'next_overs_ll_95'], latest.loc[:, 'next_overs_ul_95'] = latest[
                                                                                       'next_overs'] - scipy.stats.norm.ppf(
                .95) * (sd_next_overs / math.sqrt(len(X_train_overs))), latest['next_overs'] + scipy.stats.norm.ppf(
                .95) * (sd_next_overs / math.sqrt(len(X_train_overs)))
            latest.loc[:, 'next_runs_given_ll_95'], latest.loc[:, 'next_runs_given_ul_95'] = latest[
                                                                                                 'next_runs_given'] - scipy.stats.norm.ppf(
                .95) * (sd_next_runs_given / math.sqrt(len(X_train_runs_given))), latest[
                                                                                                 'next_runs_given'] + scipy.stats.norm.ppf(
                .95) * (sd_next_runs_given / math.sqrt(len(X_train_runs_given)))
            latest.loc[:, 'next_wkts_ll_95'], latest.loc[:, 'next_wkts_ul_95'] = latest[
                                                                                     'next_wkts'] - scipy.stats.norm.ppf(
                .95) * (sd_next_wkts / math.sqrt(len(X_train_wkts))), latest['next_wkts'] + scipy.stats.norm.ppf(
                .95) * (sd_next_wkts / math.sqrt(len(X_train_wkts)))
            models = models._append(latest)
            # print(models.columns)
            # print(player_name, " is added")

    models['next_runs_given'] = np.where(models['next_overs'] > (90 * 5),  # Considering 90 overs per day for 5 days
                                         models['next_runs_given'] / models['next_overs'] * (90 * 5),
                                         models['next_runs_given'])
    models['next_runs_given_ll_95'] = np.where(models['next_overs'] > (90 * 5),
                                               models['next_runs_given_ll_95'] / models['next_overs'] * (90 * 5),
                                               models['next_runs_given_ll_95'])
    models['next_runs_given_ul_95'] = np.where(models['next_overs'] > (90 * 5),
                                               models['next_runs_given_ul_95'] / models['next_overs'] * (90 * 5),
                                               models['next_runs_given_ul_95'])

    # Limiting next_overs to a maximum of 90 * 5 (5 days of play)
    models['next_overs'] = np.where(models['next_overs'] > (90 * 5), (90 * 5), models['next_overs'])
    models['next_overs_ll_95'] = np.where(models['next_overs_ll_95'] > (90 * 5), (90 * 5), models['next_overs_ll_95'])
    models['next_overs_ul_95'] = np.where(models['next_overs_ul_95'] > (90 * 5), (90 * 5), models['next_overs_ul_95'])

    # Adjusting next_runs based on next_balls considering longer Test innings
    models['next_runs'] = np.where(models['next_balls'] < 0, 0, models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_balls'] < 0, 0, models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_balls'] < 0, 0, models['next_runs_ul_95'])

    # Setting next_runs to a minimum of 1 for Test matches
    models['next_runs'] = np.where(models['next_runs'] < 0, 1, models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_runs_ll_95'] < 0, 1, models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_runs_ul_95'] < 0, 1, models['next_runs_ul_95'])

    # Adjusting next_runs based on next_balls if next_balls > 600 (example for Test matches)
    models['next_runs'] = np.where(models['next_balls'] > 600, models['next_runs'] / models['next_balls'] * 100,
                                   models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_balls'] > 600,
                                         models['next_runs_ll_95'] / models['next_balls'] * 100,
                                         models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_balls'] > 600,
                                         models['next_runs_ul_95'] / models['next_balls'] * 100,
                                         models['next_runs_ul_95'])

    # Limiting next_balls to a maximum of 600 (example for Test matches)
    models['next_balls'] = np.where(models['next_balls'] > 600, 600, models['next_balls'])
    models['next_balls_ll_95'] = np.where(models['next_balls_ll_95'] > 600, 600, models['next_balls_ll_95'])
    models['next_balls_ul_95'] = np.where(models['next_balls_ul_95'] > 600, 600, models['next_balls_ul_95'])

    # Setting next_balls to a minimum of 1 for Test matches
    models['next_balls'] = np.where(models['next_balls'] < 0, 1, models['next_balls'])
    models['next_balls_ll_95'] = np.where(models['next_balls_ll_95'] < 0, 1, models['next_balls_ll_95'])
    models['next_balls_ul_95'] = np.where(models['next_balls_ul_95'] < 0, 1, models['next_balls_ul_95'])

    # Setting next_wkts to a minimum of 1 for Test matches (example)
    models['next_wkts'] = np.where(models['next_wkts'] < 1, 0, models['next_wkts'])
    models['next_wkts_ll_95'] = np.where(models['next_wkts_ll_95'] < 1, 0, models['next_wkts_ll_95'])
    models['next_wkts_ul_95'] = np.where(models['next_wkts_ul_95'] < 1, 0, models['next_wkts_ul_95'])
    models['next_runs'] = round(models['next_runs'], 0)
    models['next_runs_ll_95'] = round(models['next_runs_ll_95'], 0)
    models['next_runs_ul_95'] = round(models['next_runs_ul_95'], 0)

    models['next_balls'] = round(models['next_balls'], 0)
    models['next_balls_ll_95'] = round(models['next_balls_ll_95'], 0)
    models['next_balls_ul_95'] = round(models['next_balls_ul_95'], 0)

    models['next_wkts'] = round(models['next_wkts'], 0)
    models['next_wkts_ll_95'] = round(models['next_wkts_ll_95'], 0)
    models['next_wkts_ul_95'] = round(models['next_wkts_ul_95'], 0)

    models['next_runs_given'] = round(models['next_runs_given'], 0)
    models['next_runs_given_ll_95'] = round(models['next_runs_given_ll_95'], 0)
    models['next_runs_given_ul_95'] = round(models['next_runs_given_ul_95'], 0)

    models['next_overs'] = round(models['next_overs'], 0)
    models['next_overs_ll_95'] = round(models['next_overs_ll_95'], 0)
    models['next_overs_ul_95'] = round(models['next_overs_ul_95'], 0)
    models.to_excel('modelTEST.xlsx')
    merged_df = pd.merge(models, playerData, on='Player', how='left')
    merged_df.to_excel('mergedTEST.xlsx')
    # print('successfully done')
def playing_eleven_TEST():
    merged_df = pd.read_excel('mergedTEST.xlsx')
    num_rows = len(merged_df)
    # print(len(merged_df))
    if num_rows <= 11:
        return merged_df

    batsman_data = merged_df[merged_df['Role'] == 'Batter']
    bowler_data = merged_df[merged_df['Role'] == 'Bowler']
    wicketK_data = merged_df[merged_df['Role'] == 'WK Keeper - Batter']
    allrounder_data = merged_df[merged_df['Role'] == 'Allrounder']

    batsman = batsman_data.sort_values(by='next_runs', ascending=False).iloc[:5]
    bowler = bowler_data.sort_values(by='next_wkts', ascending=False).iloc[:3]
    allrounder = allrounder_data.sort_values(by=['next_wkts', 'next_runs'], ascending=[False, False]).iloc[:2]

    if allrounder['next_runs'].iloc[0] == 0:
        allrounder = allrounder_data.sort_values(by=['next_wkts', 'next_runs'], ascending=[False, False]).iloc[1:3]

    wicketK = wicketK_data.sort_values(by='next_runs', ascending=False).iloc[:1]
    playing_eleven = pd.concat([batsman, allrounder, wicketK, bowler])
    return playing_eleven
def predict_and_result_T20(combined_players):
    final_data = pd.read_excel("T20_data_scrapped.xlsx")
    playerData = pd.read_excel("T20Image.xlsx")
    models = pd.DataFrame()
    latest = pd.DataFrame()
    players_list = combined_players
    for player_name in players_list:
        # print(player_name)
        player_data = final_data[final_data['Player'] == player_name]
        if len(player_data) > 2:
            player_new = player_data.dropna()

            # Predict next runs
            X_runs = player_new[player_new.columns[2:11]]
            y_runs = player_new[player_new.columns[21:22]]
            X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(X_runs, y_runs, random_state=123)
            ridge_runs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_runs = linear_model.Ridge(alpha=j).fit(X_train_runs, y_train_runs)
                ridge_df_runs = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_runs.score(X_train_runs, y_train_runs)),
                     'Test': pd.Series(points_runs.score(X_test_runs, y_test_runs))})
                ridge_runs = ridge_runs._append(ridge_df_runs)
            # print(ridge_runs)
            # Calculate average score
            ridge_runs['Average'] = ridge_runs[['Train', 'Test']].mean(axis=1)

            ridge_runs.sort_values(by='Average', ascending=False, inplace=True)
            k_runs = ridge_runs.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_runs = linear_model.Ridge(alpha=k_runs * 10)
            next_runs.fit(X_train_runs, y_train_runs)
            if len(X_train_runs['Runs Scored']) > 1:
                sd_next_runs = stdev(X_train_runs['Runs Scored'].astype('float'))
            else:
                # Handle empty or single-element case (assign default value, skip, etc.)
                sd_next_runs = 0.0

            # Predict next balls
            X_balls = player_new[player_new.columns[2:11]]
            y_balls = player_new[player_new.columns[22:23]]
            X_train_balls, X_test_balls, y_train_balls, y_test_balls = train_test_split(X_balls, y_balls, test_size=0.2,
                                                                                        random_state=123)
            ridge_balls = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_balls = linear_model.Ridge(alpha=j).fit(X_train_balls, y_train_balls)
                ridge_df_balls = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_balls.score(X_train_balls, y_train_balls)),
                     'Test': pd.Series(points_balls.score(X_test_balls, y_test_balls))})
                ridge_balls = ridge_balls._append(ridge_df_balls)

            # Calculate average score
            ridge_balls['Average'] = ridge_balls[['Train', 'Test']].mean(axis=1)

            ridge_balls.sort_values(by='Average', ascending=False, inplace=True)
            k_balls = ridge_balls.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_balls = linear_model.Ridge(alpha=k_balls * 10)
            next_balls.fit(X_train_balls, y_train_balls)
            if len(X_train_balls['Balls Played']) > 1:
                sd_next_balls = stdev(X_train_balls['Balls Played'].astype('float'))
            else:
                sd_next_balls = 0.0

            # Predict next overs
            X_overs = player_new[player_new.columns[11:21]]
            y_overs = player_new[player_new.columns[23:24]]
            X_train_overs, X_test_overs, y_train_overs, y_test_overs = train_test_split(X_overs, y_overs, test_size=0.2,
                                                                                        random_state=123)
            ridge_overs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_overs = linear_model.Ridge(alpha=j).fit(X_train_overs, y_train_overs)
                ridge_df_overs = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_overs.score(X_train_overs, y_train_overs)),
                     'Test': pd.Series(points_overs.score(X_test_overs, y_test_overs))})
                ridge_overs = ridge_overs._append(ridge_df_overs)

            # Calculate average score
            ridge_overs['Average'] = ridge_overs[['Train', 'Test']].mean(axis=1)

            ridge_overs.sort_values(by='Average', ascending=False, inplace=True)
            k_overs = ridge_overs.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_overs = linear_model.Ridge(alpha=k_overs * 10)
            next_overs.fit(X_train_overs, y_train_overs)
            if len(X_train_overs['Overs Bowled']) > 1:
                sd_next_overs = stdev(X_train_overs['Overs Bowled'].astype('float'))
            else:
                sd_next_overs = 0.0

            # Predict next runs given
            X_runs_given = player_new[player_new.columns[11:21]]
            y_runs_given = player_new[player_new.columns[24:25]]
            X_train_runs_given, X_test_runs_given, y_train_runs_given, y_test_runs_given = train_test_split(
                X_runs_given,
                y_runs_given,
                test_size=0.2,
                random_state=123)
            ridge_runs_given = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_runs_given = linear_model.Ridge(alpha=j).fit(X_train_runs_given, y_train_runs_given)
                ridge_df_runs_given = pd.DataFrame({'Alpha': pd.Series(j), 'Train': pd.Series(
                    points_runs_given.score(X_train_runs_given, y_train_runs_given)), 'Test': pd.Series(
                    points_runs_given.score(X_test_runs_given, y_test_runs_given))})
                ridge_runs_given = ridge_runs_given._append(ridge_df_runs_given)

            # Calculate average score
            ridge_runs_given['Average'] = ridge_runs_given[['Train', 'Test']].mean(axis=1)

            ridge_runs_given.sort_values(by='Average', ascending=False, inplace=True)
            k_runs_given = ridge_runs_given.head(1)['Alpha'].values[0]

            # Train the model with the best alpha value
            next_runs_given = linear_model.Ridge(alpha=k_runs_given * 10)
            next_runs_given.fit(X_train_runs_given, y_train_runs_given)
            if len(X_train_runs_given) > 1:
                sd_next_runs_given = stdev(X_train_runs_given['Runs Given'].astype('float'))
            else:
                sd_next_runs_given = 0.0

            X_wkts = player_new[player_new.columns[11:21]]
            y_wkts = player_new[player_new.columns[25:26]]
            X_train_wkts, X_test_wkts, y_train_wkts, y_test_wkts = train_test_split(X_wkts, y_wkts, test_size=0.2,
                                                                                    random_state=123)
            ridge_wkts = pd.DataFrame()
            # Iterate over a range of alpha values
            for j in range(0, 101):
                points_wkts = linear_model.Ridge(alpha=j).fit(X_train_wkts, y_train_wkts)
                ridge_df_wkts = pd.DataFrame(
                    {'Alpha': pd.Series(j), 'Train': pd.Series(points_wkts.score(X_train_wkts, y_train_wkts)),
                     'Test': pd.Series(points_wkts.score(X_test_wkts, y_test_wkts))})
                ridge_wkts = ridge_wkts._append(ridge_df_wkts)

            # Calculate average score
            ridge_wkts['Average'] = ridge_wkts[['Train', 'Test']].mean(axis=1)
            ridge_wkts.sort_values(by='Average', ascending=False, inplace=True)
            k_wkts = ridge_wkts.head(1)['Alpha'].values[0]
            #
            # Train the model with the best alpha value
            next_wkts = linear_model.Ridge(alpha=k_wkts * 10)
            next_wkts.fit(X_train_wkts, y_train_wkts)
            if len(X_train_wkts) > 1:
                sd_next_wkts = stdev(X_train_wkts['Wickets Taken'].astype('float'))
            else:
                sd_next_wkts = 0.0

            # Get the latest data for the player
            latest = player_data.groupby('Player').tail(1)

            latest.loc[:, 'next_runs'] = next_runs.predict(latest[latest.columns[2:11]])
            latest.loc[:, 'next_balls'] = next_balls.predict(latest[latest.columns[2:11]])
            latest.loc[:, 'next_overs'] = next_overs.predict(latest[latest.columns[11:21]])
            latest.loc[:, 'next_runs_given'] = next_runs_given.predict(latest[latest.columns[11:21]])
            latest.loc[:, 'next_wkts'] = next_wkts.predict(latest[latest.columns[11:21]])

            latest = latest.copy()
            latest.loc[:, 'next_runs_ll_95'], latest.loc[:, 'next_runs_ul_95'] = latest[
                                                                                     'next_runs'] - scipy.stats.norm.ppf(
                .95) * (sd_next_runs / math.sqrt(len(X_train_runs))), latest['next_runs'] + scipy.stats.norm.ppf(
                .95) * (sd_next_runs / math.sqrt(len(X_train_runs)))
            latest.loc[:, 'next_balls_ll_95'], latest.loc[:, 'next_balls_ul_95'] = latest[
                                                                                       'next_balls'] - scipy.stats.norm.ppf(
                .95) * (sd_next_balls / math.sqrt(len(X_train_balls))), latest['next_balls'] + scipy.stats.norm.ppf(
                .95) * (sd_next_balls / math.sqrt(len(X_train_balls)))
            latest.loc[:, 'next_overs_ll_95'], latest.loc[:, 'next_overs_ul_95'] = latest[
                                                                                       'next_overs'] - scipy.stats.norm.ppf(
                .95) * (sd_next_overs / math.sqrt(len(X_train_overs))), latest['next_overs'] + scipy.stats.norm.ppf(
                .95) * (sd_next_overs / math.sqrt(len(X_train_overs)))
            latest.loc[:, 'next_runs_given_ll_95'], latest.loc[:, 'next_runs_given_ul_95'] = latest[
                                                                                                 'next_runs_given'] - scipy.stats.norm.ppf(
                .95) * (sd_next_runs_given / math.sqrt(len(X_train_runs_given))), latest[
                                                                                                 'next_runs_given'] + scipy.stats.norm.ppf(
                .95) * (sd_next_runs_given / math.sqrt(len(X_train_runs_given)))
            latest.loc[:, 'next_wkts_ll_95'], latest.loc[:, 'next_wkts_ul_95'] = latest[
                                                                                     'next_wkts'] - scipy.stats.norm.ppf(
                .95) * (sd_next_wkts / math.sqrt(len(X_train_wkts))), latest['next_wkts'] + scipy.stats.norm.ppf(
                .95) * (sd_next_wkts / math.sqrt(len(X_train_wkts)))
            print(player_name)
            models = models._append(latest)
            # print(models.columns)
            # print(player_name, " is added")

    models['next_runs_given'] = np.where(models['next_overs'] > 4, models['next_runs_given'] / models['next_overs'] * 4,
                                         models['next_runs_given'])
    models['next_runs_given_ll_95'] = np.where(models['next_overs'] > 4,
                                               models['next_runs_given_ll_95'] / models['next_overs'] * 4,
                                               models['next_runs_given_ll_95'])
    models['next_runs_given_ul_95'] = np.where(models['next_overs'] > 4,
                                               models['next_runs_given_ul_95'] / models['next_overs'] * 4,
                                               models['next_runs_given_ul_95'])

    # Limiting next_overs to a maximum of 4
    models['next_overs'] = np.where(models['next_overs'] > 4, 4, models['next_overs'])
    models['next_overs_ll_95'] = np.where(models['next_overs_ll_95'] > 4, 4, models['next_overs_ll_95'])
    models['next_overs_ul_95'] = np.where(models['next_overs_ul_95'] > 4, 4, models['next_overs_ul_95'])

    # Adjusting next_runs based on next_balls
    models['next_runs'] = np.where(models['next_balls'] < 0, 10, models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_balls'] < 0, 12, models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_balls'] < 0, 14, models['next_runs_ul_95'])

    # Setting next_runs to a minimum of 1
    models['next_runs'] = np.where(models['next_runs'] < 0, 11, models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_runs_ll_95'] < 0, 12, models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_runs_ul_95'] < 0, 13, models['next_runs_ul_95'])

    # Adjusting next_runs based on next_balls if next_balls > 100
    models['next_runs'] = np.where(models['next_balls'] > 24, models['next_runs'] / models['next_balls'] * 24,
                                   models['next_runs'])
    models['next_runs_ll_95'] = np.where(models['next_balls'] > 24,
                                         models['next_runs_ll_95'] / models['next_balls'] * 24,
                                         models['next_runs_ll_95'])
    models['next_runs_ul_95'] = np.where(models['next_balls'] > 24,
                                         models['next_runs_ul_95'] / models['next_balls'] * 24,
                                         models['next_runs_ul_95'])

    # Limiting next_balls to a maximum of 5
    models['next_balls'] = np.where(models['next_balls'] > 24, 24, models['next_balls'])
    models['next_balls_ll_95'] = np.where(models['next_balls_ll_95'] > 24, 24, models['next_balls_ll_95'])
    models['next_balls_ul_95'] = np.where(models['next_balls_ul_95'] > 24, 24, models['next_balls_ul_95'])

    # Setting next_balls to a minimum of 1
    models['next_balls'] = np.where(models['next_balls'] < 0, 1, models['next_balls'])
    models['next_balls_ll_95'] = np.where(models['next_balls_ll_95'] < 0, 1, models['next_balls_ll_95'])
    models['next_balls_ul_95'] = np.where(models['next_balls_ul_95'] < 0, 1, models['next_balls_ul_95'])

    # Setting next_wkts to a minimum of 1
    models['next_wkts'] = np.where(models['next_wkts'] < 0, 0, models['next_wkts'])
    models['next_wkts_ll_95'] = np.where(models['next_wkts_ll_95'] < 0, 0, models['next_wkts_ll_95'])
    models['next_wkts_ul_95'] = np.where(models['next_wkts_ul_95'] < 0, 0, models['next_wkts_ul_95'])


    models['next_runs'] = round(models['next_runs'], 0)
    models['next_runs_ll_95'] = round(models['next_runs_ll_95'], 0)
    models['next_runs_ul_95'] = round(models['next_runs_ul_95'], 0)

    models['next_balls'] = round(models['next_balls'], 0)
    models['next_balls_ll_95'] = round(models['next_balls_ll_95'], 0)
    models['next_balls_ul_95'] = round(models['next_balls_ul_95'], 0)

    models['next_wkts'] = round(models['next_wkts'], 0)
    models['next_wkts_ll_95'] = round(models['next_wkts_ll_95'], 0)
    models['next_wkts_ul_95'] = round(models['next_wkts_ul_95'], 0)

    models['next_runs_given'] = round(models['next_runs_given'], 0)
    models['next_runs_given_ll_95'] = round(models['next_runs_given_ll_95'], 0)
    models['next_runs_given_ul_95'] = round(models['next_runs_given_ul_95'], 0)

    models['next_overs'] = round(models['next_overs'], 0)
    models['next_overs_ll_95'] = round(models['next_overs_ll_95'], 0)
    models['next_overs_ul_95'] = round(models['next_overs_ul_95'], 0)
    models.to_excel('modelT20.xlsx')
    merged_df = pd.merge(models, playerData, on='Player', how='left')
    merged_df.to_excel('mergedT20.xlsx')
    # print('successfully done')
def playing_eleven_T20():
    merged_df = pd.read_excel('mergedT20.xlsx')
    num_rows = len(merged_df)
    if num_rows <= 11:
        return merged_df

    batsman_data = merged_df[merged_df['Role'] == 'Batter']
    bowler_data = merged_df[merged_df['Role'] == 'Bowler']
    wicketK_data = merged_df[merged_df['Role'] == 'WK Keeper - Batter']
    allrounder_data = merged_df[merged_df['Role'] == 'Allrounder']

    batsman = batsman_data.sort_values(by='next_runs', ascending=False).iloc[:5]
    bowler = bowler_data.sort_values(by='next_wkts', ascending=False).iloc[:3]
    allrounder = allrounder_data.sort_values(by=['next_wkts', 'next_runs'], ascending=[False, False]).iloc[:2]

    if allrounder['next_runs'].iloc[0] == 0:
        allrounder = allrounder_data.sort_values(by=['next_wkts', 'next_runs'], ascending=[False, False]).iloc[1:3]

    wicketK = wicketK_data.sort_values(by='next_runs', ascending=False).iloc[:1]
    print(wicketK)
    playing_eleven = pd.concat([batsman, allrounder, wicketK, bowler])
    return playing_eleven


app = Flask(__name__)
app.secret_key = 'À²vÍ↑ÌZ↓f£fè▬íÐàÀà'
app.config["MONGO_URI"] = "mongodb+srv://Siddharth:f767Z1XZ17SJCjwm@cluster0.6kifhhz.mongodb.net/Login_details"
mongo = PyMongo(app)
db = mongo.db
bcrypt = Bcrypt(app)

# password f767Z1XZ17SJCjwm
password_regex = r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*?&]{8,}$'
def is_valid_password(password):
    return re.match(password_regex, password) is not None
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect('/login')  # Redirect to login page if not logged in
        return func(*args, **kwargs)

    return wrapper


@app.route("/")
def main():
    return render_template('main.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # print(username,password,email)
        collections = db.test_collection
        user = collections.find_one({'username': username, 'email': email})

        if user and bcrypt.check_password_hash(user['password'], password):
            # Password is correct
            session['logged_in'] = True
            session['username'] = username
            session['password'] = True
            return render_template('home.html', username=username)
        else:
            valid = 'Please enter correct credentials'
            return render_template('login.html', valid=valid)

    return render_template('login.html')


@app.route('/getstarted')
def started():
    return render_template('getstarted.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        err = ''
        name = request.form['name']
        email = request.form['email']
        username = request.form['Username']  # Ensure correct case for form field name
        password = request.form['password']
        cpassword = request.form['cpassword']

        if not is_valid_password(password):
            err += 'Password must be at least 8 characters long and contain at least one letter and one digit.'
            return render_template('signup.html', err=err)


        if password != cpassword:
            err += 'Please enter correct password'
        collection = db.test_collection

        # Inserting data into the collection
        data_to_insert = {
            'name': name,
            'email': email,
            'username': username,
            'password': bcrypt.generate_password_hash(password).decode('utf-8')
        }
        if err != '':
            return redirect(url_for('signup', err=err))
        else:
            collection.insert_one(data_to_insert)
            return render_template('main.html')

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('main'))


@app.route("/home", methods=["GET", "POST"])
@login_required
def home():
    # Check if user is logged in
    if 'username' in session:
        username = session['username']
        # print(f"The logged-in username is: {username}")
        return render_template('home.html', username=username)
    else:
        return redirect(url_for('login'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/submitFeedback', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        feedback = request.form['message']
        collection = db.feedback

        # Inserting data into the collection
        data_to_insert = {
            'name': name,
            'email': email,
            'message': feedback
        }
        value = ''
        try:
            # Attempt to insert data into the collection
            collection.insert_one(data_to_insert)
            value = 'feedback posted successfully !'
            return render_template('feedback.html', value=value)
        except Exception as e:
            value = 'Feedback not sent please try again'
            return render_template('feedback.html', value=value)


@app.route('/feedback')
@login_required
def feedback():
    return render_template('feedback.html')


@app.route('/getusers', methods=['POST'])
def getuser():
    if request.method == 'POST':
        # Fetch all documents from MongoDB collection
        collection = db['test_collection']
        all_documents = list(collection.find())
        return render_template('getusers.html', all_documents=all_documents)
    return 'Invalid request'


@app.route('/update_players_IPL', methods=['POST'])
@login_required
def update_players_IPL():
    if request.method == 'POST':
        data = request.get_json()
        if data and 'players' in data:
            IPL_players.clear()
            for player in data['players']:
                IPL_players.append(player.get('name'))
        print('these are updated player of UPDate function \n', IPL_players)
        return "success"


@app.route('/update_players_ODI', methods=['POST'])
@login_required
def update_players_ODI():
    if request.method == 'POST':
        data = request.get_json()
        if data and 'players' in data:
            ODI_players.clear()
            for player in data['players']:
                ODI_players.append(player.get('name'))
        print('these are updated player \n', ODI_players)
        return "success"


@app.route('/update_players_TEST', methods=['POST'])
@login_required
def update_players_TEST():
    if request.method == 'POST':
        data = request.get_json()
        if data and 'players' in data:
            TEST_players.clear()
            for player in data['players']:
                TEST_players.append(player.get('name'))
        print('these are updated player \n', TEST_players)
        return "success"


@app.route('/update_players_T20', methods=['POST'])
@login_required
def update_players_T20():
    if request.method == 'POST':
        data = request.get_json()
        if data and 'players' in data:
            T20_players.clear()
            for player in data['players']:
                T20_players.append(player.get('name'))
        print('these are updated player \n', T20_players)
        return "success"


@app.route("/ipl")
@login_required
def ipl():
    merged_headers = IPLdata.columns.get_level_values(0).unique()
    cleaned_headers = [header for header in merged_headers if 'Unnamed' not in header]
    return render_template('ipl.html', cleaned_headers=cleaned_headers)


@app.route('/predictIPL', methods=['GET', 'POST'])
async def predictIPL():
    if request.method == 'POST':
        print('posted request')
        selected_team1 = request.form.get('selected_team1')
        selected_team2 = request.form.get('selected_team2')
        team1_data = IPLdata.loc[:, (selected_team1, ["Name", "Role", "Image"])]
        team2_data = IPLdata.loc[:, (selected_team2, ["Name", "Role", "Image"])]
        return render_template('predictIPL.html', team1_data=team1_data, team2_data=team2_data)
    else:
        return render_template('ipl.html')


@app.route('/predictedIPL')
@login_required
def predictedIPL():
    print('The ipl players of predictedIPL\n', IPL_players)
    predict_and_result_IPL(IPL_players)
    eleven = playing_eleven_IPL()
    return render_template('predictedIPL.html', eleven=eleven)


@app.route("/odi")
@login_required
def odi():
    merged_headers = ODIdata.columns.get_level_values(0).unique()
    cleaned_headers = [header for header in merged_headers if 'Unnamed' not in header]
    return render_template('odi.html', cleaned_headers=cleaned_headers)


@app.route("/predictODI", methods=['GET', 'POST'])
def predictODI():
    if request.method == 'POST':
        selected_team1 = request.form.get('selected_team1')
        selected_team2 = request.form.get('selected_team2')
        team1_data = ODIdata.loc[:, (selected_team1, ["Name", "Role", "Image"])]
        team2_data = ODIdata.loc[:, (selected_team2, ["Name", "Role", "Image"])]
        return render_template('predictODI.html', team1_data=team1_data, team2_data=team2_data)
    else:
        return render_template('odi.html')


@app.route('/predictedODI')
@login_required
def predictedODI():
    print('The player of predicted ODI')
    print(ODI_players)
    predict_and_result_ODI(ODI_players)
    eleven = playing_eleven_ODI()
    return render_template('predictedODI.html', eleven=eleven)


@app.route("/test")
@login_required
def test():
    merged_headers = TESTdata.columns.get_level_values(0).unique()
    cleaned_headers = [header for header in merged_headers if 'Unnamed' not in header]
    return render_template('test.html', cleaned_headers=cleaned_headers)


@app.route("/predictTEST", methods=['GET', 'POST'])
def predictTEST():
    if request.method == 'POST':
        selected_team1 = request.form.get('selected_team1')
        selected_team2 = request.form.get('selected_team2')
        team1_data = TESTdata.loc[:, (selected_team1, ["Name", "Role", "Image"])]
        team2_data = TESTdata.loc[:, (selected_team2, ["Name", "Role", "Image"])]
        return render_template('predictTEST.html', team1_data=team1_data, team2_data=team2_data)
    else:
        return render_template('test.html')


@app.route('/predictedTEST')
@login_required
def predictedTEST():
    print('The player of predicted Test')
    print(TEST_players)
    predict_and_result_TEST(TEST_players)
    eleven = playing_eleven_TEST()
    return render_template('predictedTEST.html', eleven=eleven)


@app.route("/t20")
@login_required
def t20():
    merged_headers = T20data.columns.get_level_values(0).unique()
    cleaned_headers = [header for header in merged_headers if 'Unnamed' not in header]
    return render_template('t20.html', cleaned_headers=cleaned_headers)


@app.route("/predictT20", methods=['GET', 'POST'])
def predictT20():
    if request.method == 'POST':
        selected_team1 = request.form.get('selected_team1')
        selected_team2 = request.form.get('selected_team2')
        team1_data = T20data.loc[:, (selected_team1, ["Name", "Role", "Image"])]
        team2_data = T20data.loc[:, (selected_team2, ["Name", "Role", "Image"])]
        print(team2_data)
        print(team1_data)
        return render_template('predictT20.html', team1_data=team1_data, team2_data=team2_data)
    else:
        return render_template('t20.html')


@app.route('/predictedT20')
@login_required
def predictedT20():
    print('The player of predicted Test')
    print(T20_players)
    predict_and_result_T20(T20_players)
    print('predict_and_result for T20 is done')
    eleven = playing_eleven_T20()
    return render_template('predictedT20.html',eleven=eleven)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
