from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

URL = "https://medi-assist-ai-rag-v2.streamlit.app"


options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)

try:
    print("Opening Streamlit app...")
    driver.get(URL)

    try:
        button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(., 'Yes, get this app back up!')]")
            )
        )

        print("App was sleeping. Clicking wake-up button...")
        button.click()

    except TimeoutException:
        print("App is already awake.")

finally:
    driver.quit()