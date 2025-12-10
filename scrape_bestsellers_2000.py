import time
import json
import logging
import csv
import random
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global settings
TARGET_PRODUCTS = 2000
total_products_scraped = 0


def setup_driver():
    """Setup Chrome WebDriver with stealth options"""
    chrome_options = Options()

    # Stealth mode settings
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

    # Uncomment to run headless
    # chrome_options.add_argument('--headless=new')

    # Initialize driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Execute script to prevent detection
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })

    logger.info("✓ Chrome WebDriver initialized")
    return driver


def human_like_scroll(driver):
    """Scroll page in a human-like manner"""
    try:
        # Get page height
        last_height = driver.execute_script("return document.body.scrollHeight")

        # Scroll in chunks
        scroll_pause_time = random.uniform(0.5, 1.5)
        scroll_increment = random.randint(300, 500)

        current_position = 0
        while current_position < last_height:
            # Scroll down
            current_position += scroll_increment
            driver.execute_script(f"window.scrollTo(0, {current_position});")
            time.sleep(scroll_pause_time)

            # Check if new content loaded
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height > last_height:
                last_height = new_height

        # Scroll back to top
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

    except Exception as e:
        logger.warning(f"Scroll error: {str(e)}")


def extract_products_from_page(driver, category_name):
    """Extract all products from current page"""
    products = []
    seen_asins = set()

    try:
        # Wait for products to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-asin], .zg-item"))
        )

        # Scroll to load all content
        human_like_scroll(driver)

        # Find all product containers
        product_elements = driver.find_elements(By.CSS_SELECTOR, "[data-asin]:not([data-asin='']), .zg-grid-general-faceout")

        logger.info(f"  Found {len(product_elements)} product elements")

        for element in product_elements:
            try:
                product = {}

                # Extract ASIN (product ID)
                asin = element.get_attribute('data-asin')
                if not asin or asin in seen_asins:
                    continue

                seen_asins.add(asin)
                product['product_id'] = asin
                product['category'] = category_name
                product['scraped_at'] = datetime.now().isoformat()

                # Extract product name
                try:
                    name_elem = element.find_element(By.CSS_SELECTOR, "a.a-link-normal span div, ._cDEzb_p13n-sc-css-line-clamp-3_g3dy1, .p13n-sc-truncate")
                    product['product_name'] = name_elem.text.strip()
                except:
                    try:
                        name_elem = element.find_element(By.CSS_SELECTOR, "a.a-link-normal")
                        product['product_name'] = name_elem.get_attribute('aria-label') or name_elem.text.strip()
                    except:
                        product['product_name'] = ""

                # Extract rank
                try:
                    rank_elem = element.find_element(By.CSS_SELECTOR, ".zg-bdg-text, .zg-badge-text, span.a-badge-text")
                    product['rank'] = rank_elem.text.strip()
                except:
                    product['rank'] = ""

                # Extract price
                try:
                    price_elem = element.find_element(By.CSS_SELECTOR, "._cDEzb_p13n-sc-price_3mJ9Z, .p13n-sc-price, span.a-color-price, .a-price .a-offscreen")
                    product['discounted_price'] = price_elem.text.strip()
                except:
                    product['discounted_price'] = ""

                # Extract rating
                try:
                    rating_elem = element.find_element(By.CSS_SELECTOR, "span.a-icon-alt, i.a-icon-star-small")
                    product['rating'] = rating_elem.get_attribute('textContent') or rating_elem.text.strip()
                except:
                    product['rating'] = ""

                # Extract review count
                try:
                    review_elem = element.find_element(By.CSS_SELECTOR, "span.a-size-small")
                    product['review_count'] = review_elem.text.strip()
                except:
                    product['review_count'] = ""

                # Extract product link
                try:
                    link_elem = element.find_element(By.CSS_SELECTOR, "a.a-link-normal[href*='/dp/']")
                    href = link_elem.get_attribute('href')
                    product['product_link'] = href if href.startswith('http') else f"https://www.amazon.com{href}"
                except:
                    product['product_link'] = f"https://www.amazon.com/dp/{asin}"

                # Extract image
                try:
                    img_elem = element.find_element(By.CSS_SELECTOR, "img.a-dynamic-image, img")
                    product['image_url'] = img_elem.get_attribute('src')
                except:
                    product['image_url'] = ""

                # Check Prime
                try:
                    element.find_element(By.CSS_SELECTOR, "i.a-icon-prime")
                    product['is_prime'] = "Yes"
                except:
                    product['is_prime'] = "No"

                products.append(product)

            except Exception as e:
                continue

        logger.info(f"  ✓ Extracted {len(products)} products")
        return products

    except TimeoutException:
        logger.warning("  ✗ Timeout waiting for products to load")
        return []
    except Exception as e:
        logger.error(f"  ✗ Error extracting products: {str(e)}")
        return []


def scrape_category(driver, category, max_pages=2):
    """Scrape products from a category with pagination"""
    all_products = []
    category_name = category['name']
    base_url = category['url'].rstrip('/')

    for page_num in range(1, max_pages + 1):
        try:
            # Construct URL
            if page_num == 1:
                page_url = base_url
            else:
                page_url = f"{base_url}/ref=zg_bs_pg_{page_num}?_encoding=UTF8&pg={page_num}"

            logger.info(f"Scraping {category_name} - Page {page_num}/{max_pages}")
            logger.info(f"  URL: {page_url}")

            # Navigate to page
            driver.get(page_url)

            # Random delay to appear human
            time.sleep(random.uniform(3, 5))

            # Extract products
            page_products = extract_products_from_page(driver, category_name)

            if not page_products:
                logger.warning(f"  No products found on page {page_num}, stopping pagination")
                break

            all_products.extend(page_products)
            logger.info(f"  ✓ Page {page_num} complete - Total: {len(all_products)} products")

            # Check if there's a next page
            if page_num < max_pages:
                try:
                    # Look for next page button
                    next_button = driver.find_element(By.CSS_SELECTOR, "li.a-last a, .a-pagination .a-last a")
                    if not next_button.is_enabled():
                        logger.info("  → No more pages available")
                        break
                except NoSuchElementException:
                    logger.info("  → Next page button not found")
                    break

                # Random delay between pages
                delay = random.uniform(3, 6)
                logger.info(f"  → Waiting {delay:.1f}s before next page...")
                time.sleep(delay)

        except Exception as e:
            logger.error(f"  ✗ Error on page {page_num}: {str(e)}")
            break

    return all_products


def export_to_csv(products_dict, filename='amazon_bestsellers_2000.csv'):
    """Export scraped products to CSV file"""

    # Flatten products from all categories
    all_products = []
    for category, products in products_dict.items():
        all_products.extend(products)

    if not all_products:
        logger.warning("No products to export!")
        return 0

    # Define CSV columns
    fieldnames = [
        'product_id', 'product_name', 'category', 'rank',
        'discounted_price', 'rating', 'review_count',
        'is_prime', 'product_link', 'image_url', 'scraped_at'
    ]

    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_products)

    logger.info(f"✓ Exported {len(all_products)} products to {filename}")
    return len(all_products)


def main():
    """Main scraping function"""
    global total_products_scraped

    # All Amazon categories
    all_categories = [
        {"name": "Electronics", "url": "https://www.amazon.com/Best-Sellers-Electronics/zgbs/electronics/"},
        {"name": "Books", "url": "https://www.amazon.com/Best-Sellers-Books/zgbs/books/"},
        {"name": "Clothing, Shoes & Jewelry", "url": "https://www.amazon.com/Best-Sellers-Clothing-Shoes-Jewelry/zgbs/fashion/"},
        {"name": "Home & Kitchen", "url": "https://www.amazon.com/Best-Sellers-Home-Kitchen/zgbs/home-garden/"},
        {"name": "Toys & Games", "url": "https://www.amazon.com/Best-Sellers-Toys-Games/zgbs/toys-and-games/"},
        {"name": "Beauty & Personal Care", "url": "https://www.amazon.com/Best-Sellers-Beauty/zgbs/beauty/"},
        {"name": "Health & Household", "url": "https://www.amazon.com/Best-Sellers-Health-Personal-Care/zgbs/hpc/"},
        {"name": "Sports & Outdoors", "url": "https://www.amazon.com/Best-Sellers-Sports-Outdoors/zgbs/sporting-goods/"},
        {"name": "Pet Supplies", "url": "https://www.amazon.com/Best-Sellers-Pet-Supplies/zgbs/pet-supplies/"},
        {"name": "Automotive", "url": "https://www.amazon.com/Best-Sellers-Automotive/zgbs/automotive/"},
        {"name": "Tools & Home Improvement", "url": "https://www.amazon.com/Best-Sellers-Home-Improvement/zgbs/hi/"},
        {"name": "Office Products", "url": "https://www.amazon.com/Best-Sellers-Office-Products/zgbs/office-products/"},
    ]

    driver = None

    try:
        logger.info("="*80)
        logger.info(f"🚀 Starting Amazon Bestsellers Scraper - Target: {TARGET_PRODUCTS} products")
        logger.info("="*80)

        # Setup driver
        driver = setup_driver()

        all_results = {}

        # Scrape categories
        for category in all_categories:
            if total_products_scraped >= TARGET_PRODUCTS:
                logger.info(f"\n✓ Target reached: {total_products_scraped} products!")
                break

            logger.info(f"\n📦 Category: {category['name']}")
            logger.info(f"   Current total: {total_products_scraped}/{TARGET_PRODUCTS}")

            # Scrape category (2 pages max)
            products = scrape_category(driver, category, max_pages=2)

            if products:
                all_results[category['name']] = products
                total_products_scraped += len(products)
                logger.info(f"   ✓ Category total: {len(products)} products")
                logger.info(f"   ✓ Grand total: {total_products_scraped}/{TARGET_PRODUCTS}")

            # Delay between categories
            if total_products_scraped < TARGET_PRODUCTS:
                delay = random.uniform(5, 10)
                logger.info(f"   → Waiting {delay:.1f}s before next category...")
                time.sleep(delay)

        # Export results
        logger.info("\n" + "="*80)
        logger.info("💾 Exporting data to CSV...")
        logger.info("="*80)

        csv_count = export_to_csv(all_results, 'amazon_bestsellers_2000.csv')

        # Also save JSON
        with open("amazon_bestsellers_2000.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved JSON: amazon_bestsellers_2000.json")

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("📊 SCRAPING COMPLETE - SUMMARY")
        logger.info("="*80)
        logger.info(f"✓ Total products scraped: {total_products_scraped}")
        logger.info(f"✓ Categories scraped: {len(all_results)}")
        logger.info(f"✓ CSV file: amazon_bestsellers_2000.csv ({csv_count} rows)")
        logger.info(f"✓ JSON file: amazon_bestsellers_2000.json")

        # Category breakdown
        logger.info("\n📋 Breakdown by category:")
        for cat_name, products in all_results.items():
            logger.info(f"   • {cat_name}: {len(products)} products")

        logger.info("\n🎉 Scraping completed successfully!")

    except Exception as e:
        logger.error(f"\n❌ Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        if driver:
            logger.info("\n🔒 Closing browser...")
            driver.quit()


if __name__ == "__main__":
    main()
