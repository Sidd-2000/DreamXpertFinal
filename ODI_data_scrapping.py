import math
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
import time

Teamname = pd.read_excel('ODI_Cricket_players.xlsx', header=[0, 1])
merged_headers = Teamname.columns.get_level_values(0).unique()
player = []
print(merged_headers[1:])
for i in merged_headers[1:]:
    for j in Teamname[(i, 'Name')]:
        if j == 'NaN' or j is None:
            continue
        else:
            player.append(j)

player = [x for x in player if not (math.isnan(x) if isinstance(x, float) else False)]
driver = webdriver.Firefox()
players = player
print(player)
print(len(player))
final_data = pd.DataFrame()  # Final dataframe to store

for i in players[0:]:
    print(i)
    # Accessing the web page for the current player's stats
    driver.get("http://www.cricmetric.com/playerstats.py?player={}&role=all&format=ODI&groupby=match&playerStatsFilters=on&start_date=2015-01-01&end_date=2024-10-07&start_over=0&end_over=9999".format(i.replace(' ', '+')))
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, 1080)")
    driver.maximize_window()
    time.sleep(3)

    try:
        batting_table = driver.find_element(By.XPATH, '/html/body/div[1]/div/div[1]/div[2]/div[2]/div/div[1]/div/table')
        bat = batting_table.text
        stats = pd.DataFrame(bat.split('\n'))[0].str.split(' ', expand=True)[0:-1]
        stats.columns = stats.iloc[0]
        stats = stats[1:]
        del stats['%']
        stats = stats[['Match', 'Runs', 'Balls', 'Outs', 'SR',
                       '50', '100', '4s', '6s', 'Dot']]
        stats.columns = ['Match', 'Runs Scored', 'Balls Played',
                         'Out', 'Bat SR', '50', '100', '4s Scored',
                         '6s Scored', 'Bat Dot%']

        # Switching to bowling stats tab
        bowling_tab = driver.find_element(By.XPATH, '//*[@id="ODI-Bowling-tab"]')
        bowling_tab.click()
        time.sleep(5)

        # Extracting bowling stats of the player
        bowling_table = driver.find_element(By.XPATH,
                                            '/html/body/div[1]/div/div[1]/div[2]/div[2]/div/div[1]/div/table')
        bowl = bowling_table.text
        stats2 = pd.DataFrame(bowl.split('\n'))[0].str.split(' ', expand=True)[0:-1]
        stats2.columns = stats2.iloc[0]
        stats2 = stats2[1:]
        stats2 = stats2[['Match', 'Overs', 'Runs', 'Wickets', 'Econ',
                         'Avg', 'SR', '5W', '4s', '6s', 'Dot%']]
        stats2.columns = ['Match', 'Overs Bowled', 'Runs Given',
                          'Wickets Taken', 'Econ', 'Bowl Avg',
                          'Bowl SR', '5W', '4s Given', '6s Given',
                          'Bowl Dot%']
    except:
        # If stats for the current player are not found,
        # create an empty dataframe
        stats2 = pd.DataFrame({'Match': pd.Series(stats['Match'][0:1]),
                               'Overs Bowled': [0], 'Runs Given': [0],
                               'Wickets Taken': [0], 'Econ': [0],
                               'Bowl Avg': [0], 'Bowl SR': [0], '5W': [0],
                               '4s Given': [0], '6s Given': [0],
                               'Bowl Dot%': [0]})
    merged_stats = pd.merge(stats, stats2, on='Match', how='outer').fillna(0)
    merged_stats = merged_stats.sort_values(by=['Match'])

    # Create lagged variables for future performance prediction
    merged_stats.insert(loc=0, column='Player', value=i)
    merged_stats['next_runs'] = merged_stats['Runs Scored'].shift(-1)
    merged_stats['next_balls'] = merged_stats['Balls Played'].shift(-1)
    merged_stats['next_overs'] = merged_stats['Overs Bowled'].shift(-1)
    merged_stats['next_runs_given'] = merged_stats['Runs Given'].shift(-1)
    merged_stats['next_wkts'] = merged_stats['Wickets Taken'].shift(-1)
    final_data = final_data._append(merged_stats)

final_data = final_data[final_data['Match'] != 0]

final_data['Bowl Avg'] = np.where(final_data['Bowl Avg'] == '-',
                                  0, final_data['Bowl Avg'])
final_data['Bowl SR'] = np.where(final_data['Bowl SR'] == '-',
                                 0, final_data['Bowl SR'])
final_data = final_data[['Player', 'Match', 'Runs Scored',
                         'Balls Played', 'Out', 'Bat SR',
                         '50', '100', '4s Scored',
                         '6s Scored', 'Bat Dot%',
                         'Overs Bowled', 'Runs Given',
                         'Wickets Taken', 'Econ',
                         'Bowl Avg', 'Bowl SR', '5W',
                         '4s Given', '6s Given',
                         'Bowl Dot%', 'next_runs',
                         'next_balls', 'next_overs',
                         'next_runs_given', 'next_wkts']]
final_data = final_data.replace('-', 0)

final_data.to_excel('ODI_data_scrapped.xlsx', index=False)
