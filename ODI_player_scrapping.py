import time

import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

driver = webdriver.Firefox()

web = ["https://crictoday.com/cricket/series/afghanistan-odi-squad/",
       "https://crictoday.com/cricket/series/australia-odi-squad/",
       "https://crictoday.com/cricket/series/bangladesh-odi-squad-2023/",
       "https://crictoday.com/cricket/series/england-odi-squad-2023/",
       "https://crictoday.com/cricket/series/india-odi-squad/",
       "https://crictoday.com/cricket/series/ireland-odi-squad/",
       "https://crictoday.com/cricket/series/new-zealand-odi-squad/",
       "https://crictoday.com/cricket/series/pakistan-odi-squad/",
       "https://crictoday.com/cricket/series/south-africa-odi-squad/",
       "https://crictoday.com/cricket/series/sri-lanka-odi-squad/",
       "https://crictoday.com/cricket/series/west-indies-odi-team/",
       "https://crictoday.com/cricket/series/zimbabwe-odi-squad/"]

teams = [
    'Afghanistan', 'Australia', 'Bangladesh', 'England', 'India',
    'Ireland', 'New Zealand', 'Pakistan', 'South Africa', 'Sri Lanka', 'West Indies', 'Zimbabwe']

data_list = []

for url, team in zip(web, teams):
    driver.get(url)
    time.sleep(2)
    table = driver.find_element(By.XPATH, "/html/body/div[2]/div/section/main/div[2]/article/div/figure/table/tbody")
    rows = table.find_elements(By.TAG_NAME, 'tr')

    team_players = []
    team_data = {'Name': [], 'Role': [], 'Image': []}

    for row in rows:
        # Extracting columns from each row
        columns = row.find_elements(By.TAG_NAME, 'td')

        # Assuming the name is in the first column, role in the second column
        player_name = columns[0].text.strip()
        player_role = columns[1].text.strip()

        # Append the data to the list for the current team
        team_data['Name'].append(player_name)
        team_data['Role'].append(player_role)

    # Add a placeholder for 'Image' column
    team_data['Image'] = [''] * len(team_data['Name'])

    team_df = pd.DataFrame(team_data)
    team_df.columns = pd.MultiIndex.from_tuples([(team, "Name"), (team, "Role"), (team, 'Image')])
    data_list.append(team_df)

final_data = pd.concat(data_list, axis=1)
driver.quit()
print(final_data)

final_data.to_excel('ODI_Cricket_players.xlsx', index=True)