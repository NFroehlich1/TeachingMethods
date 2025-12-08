from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import datetime

def run_scraper():
    print(f"Starting scraper at {datetime.datetime.now()}")

    # Setup Chrome options for Headless mode (required for servers/CI)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # chrome_options.add_argument("--disable-gpu") # Sometimes helpful

    try:
        # Initialize WebDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # --- YOUR SCRAPING LOGIC HERE ---
        print("Navigating to example.com...")
        driver.get("https://example.com")
        
        title = driver.title
        print(f"Page title: {title}")
        
        # Example: Find an element
        # element = driver.find_element(By.TAG_NAME, "h1")
        # print(f"Heading: {element.text}")
        
        # Simulate some work
        time.sleep(2)
        
        print("Scraping finished successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        if 'driver' in locals():
            driver.quit()
            print("Driver closed.")

if __name__ == "__main__":
    run_scraper()

