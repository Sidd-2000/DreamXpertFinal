import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

image_url = []
driver = webdriver.Firefox()
data = pd.read_excel('TestPlayersRole.xlsx')
data2 = pd.DataFrame()
name = data['Player']
Name = [x for x in name if x == x]

for i in Name:
    driver.get('https://www.google.com/search?q=passport+size+photo+of+{}&sca_esv=594411916&tbm=isch&source=lnms&sa=X&ved=2ahUKEwiIuICOvbSDAxV_wTgGHVlhB4EQ_AUoAXoECAQQAw&biw=1280&bih=689&dpr=1'.format(i.replace(' ','+')))
    time.sleep(2)
    div = driver.find_element(By.XPATH,'/html/body/div[2]/c-wiz/div[3]/div[1]/div/div/div/div/div[1]')
    Img = div.find_elements(By.TAG_NAME,'img')
    image_url.append(Img[0].get_attribute('src'))
    print(i)

driver.quit()
data2['Image'] = image_url
data2.to_excel('TESTImage.xlsx', index=False)