import math
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

web = ['https://www.iplt20.com/teams/mumbai-indians', 'https://www.iplt20.com/teams/punjab-kings',
       'https://www.iplt20.com/teams/rajasthan-royals',
       'https://www.iplt20.com/teams/royal-challengers-bangalore',
       'https://www.iplt20.com/teams/sunrisers-hyderabad',
       'https://www.iplt20.com/teams/chennai-super-kings',
       'https://www.iplt20.com/teams/delhi-capitals',
       'https://www.iplt20.com/teams/gujarat-titans',
       'https://www.iplt20.com/teams/kolkata-knight-riders',
       'https://www.iplt20.com/teams/lucknow-super-giants']
teams = ['mumbai-indians', 'punjab-kings', 'rajasthan-royals', 'royal-challengers-bangalore', 'sunrisers-hyderabad',
         'chennai-super-kings', 'delhi-capitals', 'gujarat-titans',
         'kolkata-knight-riders', 'lucknow-super-giants']

driver = webdriver.Firefox()
data_list = []

# ...
# ...

for url, team in zip(web, teams):
    driver.get(url)
    time.sleep(2)

    # Modified XPath to select the correct table
    table = driver.find_element(By.XPATH, "/html/body/div[3]/div/div")
    print(table)

    name = []
    role = []

    # Use find_elements to locate the list items within the table
    players = table.find_elements(By.TAG_NAME, 'li')
    team_players = []

    for player in players:
        # Using try-except to avoid NoSuchElementException
        try:
            name_element = player.find_elements(By.CLASS_NAME, 'ih-p-name')
            for i in name_element:
                name.append(i.text)
            role_element = player.find_elements(By.TAG_NAME, 'span')
            for i in role_element:
                role.append(i.text)
        except:
            continue

    role = [x for x in role if x != '']

    # Create a dictionary for each team
    team_data = {'Name': name, 'Role': role, 'Image': [''] * len(name)}

    # Create a DataFrame for each team
    team_df = pd.DataFrame(team_data)

    # Add a multi-level column with the upper header as the team name
    team_df.columns = pd.MultiIndex.from_tuples([(team, "Name"), (team, "Role"), (team, 'Image')])

    # Append the DataFrame to the data_list
    data_list.append(team_df)
    print(teams)

# Concatenate all team DataFrames into a single DataFrame
final_data = pd.concat(data_list, axis=1)

# Save the DataFrame to an Excel file
final_data.to_excel('IPL_Cricket_players.xlsx', index=True)

driver.quit()

