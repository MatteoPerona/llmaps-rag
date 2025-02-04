import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
from datetime import datetime

class GroceryStoreScraper:
    # Store configurations
    STORES = [
        {
            'name': 'ralphs',
            'url': 'https://www.ralphs.com/stores/search',
            'search_id': 'SearchBar-input'
        },
        {
            'name': 'vons', 
            'url': 'https://local.vons.com/',
            'search_id': 'store-search-input'
        }
    ]

    def __init__(self, location):
        """Initialize the scraper with a location."""
        self.location = location
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--headless')  # Run in headless mode
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        # Enable PDF printing
        self.options.add_argument('--print-to-pdf')
        self.driver = None
        
    def start_driver(self):
        """Start the Chrome driver."""
        self.driver = webdriver.Chrome(options=self.options)
        
    def close_driver(self):
        """Close the Chrome driver."""
        if self.driver:
            self.driver.quit()
            
    def save_page_as_pdf(self, store_name):
        """Save the current page as PDF."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        location_slug = self.location.replace(' ', '_').replace(',', '').lower()
        filename = f'raw-documents/{store_name}_{location_slug}_{timestamp}.pdf'
        
        # Use Chrome's PDF printing capability
        pdf = self.driver.execute_cdp_cmd("Page.printToPDF", {
            "printBackground": True,
            "paperWidth": 8.27,  # A4 width in inches
            "paperHeight": 11.7,  # A4 height in inches
        })
        
        os.makedirs('raw-documents', exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(bytes(pdf['data'], 'base64'))
            
        print(f"Saved PDF to {filename}")

    def scrape_store(self, store_config):
        """Generic function to scrape any configured store."""
        try:
            print(f"Saving {store_config['name']} page for {self.location}...")
            self.driver.get(store_config['url'])
            
            # Wait for and find the search input
            search_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, store_config['search_id']))
            )
            
            # Enter location and submit
            search_input.clear()
            search_input.send_keys(self.location)
            search_input.send_keys(Keys.RETURN)
            
            # Wait for store results to load
            time.sleep(5)
            
            # Save the page as PDF
            self.save_page_as_pdf(store_config['name'])
            
        except Exception as e:
            print(f"Error saving {store_config['name']} page: {e}")

    def scrape_all_stores(self):
        """Save PDFs from all supported stores."""
        try:
            self.start_driver()
            
            # Iterate through all configured stores
            for store in self.STORES:
                self.scrape_store(store)
                
        finally:
            self.close_driver()

def main():
    # Get location from command line
    import argparse
    parser = argparse.ArgumentParser(description='Save grocery store pages as PDFs for a location')
    parser.add_argument('location', help='Location to search for stores (e.g. "La Jolla, CA")')
    args = parser.parse_args()
    
    # Initialize and run scraper
    scraper = GroceryStoreScraper(args.location)
    scraper.scrape_all_stores()
    
    print(f"\nFinished saving store pages for {args.location}")

if __name__ == "__main__":
    main()