import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By


driver = webdriver.Firefox()
data = pd.read_excel('T20_Cricket_players.xlsx',header=[0,1])
merged_headers = data.columns.get_level_values(0).unique()
cleaned_headers = [header for header in merged_headers if 'Unnamed' not in header]
playername = []
role = []

for i in range(len(cleaned_headers)):
    playername.append(data[(cleaned_headers[i], 'Name')])
    role.append(data[(cleaned_headers[i], 'Role')])

playername = [data[(header, 'Name')].dropna() for header in cleaned_headers]
role = [data[(header, 'Role')].dropna() for header in cleaned_headers]

new_data = pd.DataFrame({
    'Player': pd.concat(playername, ignore_index=True),
    'Role': pd.concat(role, ignore_index=True)
})


image_url = []
Name = new_data['Player']
for i in Name:
    driver.get('https://www.google.com/search?q=passport+size+photo+of+{}+cricketer&sca_esv=594411916&tbm=isch&source=lnms&sa=X&ved=2ahUKEwiIuICOvbSDAxV_wTgGHVlhB4EQ_AUoAXoECAQQAw&biw=1280&bih=689&dpr=1'.format(i.replace(' ','+')))
    time.sleep(2)
    div = driver.find_element(By.XPATH,'/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]')
    Img = div.find_elements(By.TAG_NAME,'img')
    image_url.append(Img[0].get_attribute('src'))
    print(i)

driver.quit()
data2 = pd.DataFrame({
    'Player':new_data['Player'],
    'Role':new_data['Role'],
    'Image': image_url
})
data2.to_excel('T20Image.xlsx', index=False)