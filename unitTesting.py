import time
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pandas as pd
import numpy as np
from flask_pymongo import PyMongo
from functools import wraps
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from statistics import stdev
import math
import scipy

results = []
combined_players =['ROHIT SHARMA','DEWALD BREVIS','SURYAKUMAR YADAV']
def predict_and_result_IPL(combined_players):
    final_data = pd.read_excel("IPL_data_scrapped.xlsx")
    playerData = pd.read_excel("IPLImage.xlsx")
    models = pd.DataFrame()
    latest = pd.DataFrame()
    players_list = combined_players
    # print('This is player list',players_list)
    for player_name in players_list:
        # print('player name player',player_name)
        print(player_name)
        player_data = final_data[final_data['Player'] == player_name]
        # print('Data of the player is', player_data)
        # print("\n", player_data['Player'])
        if len(player_data) > 2:
            player_new = player_data.dropna()
            # Predict next runs
            X_runs = player_new[player_new.columns[2:11]]
            y_runs = player_new[player_new.columns[21:22]]
            X_train_runs, X_test_runs, y_train_runs, y_test_runs = train_test_split(X_runs, y_runs, random_state=123)
            ridge_runs = pd.DataFrame()

            # Iterate over a range of alpha values
            for j in range(0, 101):
                model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust parameters as needed
                model.fit(X_train_runs, y_train_runs)

                # Calculate MSE and MAE
                predictions_runs = model.predict(X_test_runs)
                mse_runs = mean_squared_error(y_test_runs, predictions_runs)
                mae_runs = mean_absolute_error(y_test_runs, predictions_runs)

                # Print or store MSE and MAE for each iteration
                print(f'Iteration {j + 1} - Alpha: {j}, MSE: {mse_runs}, MAE: {mae_runs}')

                # Store results in the list
                results.append({'Iteration': j + 1, 'Alpha': j, 'MSE': mse_runs, 'MAE': mae_runs})
