import os
import requests
import json
import re
from PIL import Image
from io import BytesIO
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uro_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingConfig:
    """Configuration for the scraper"""
    output_folder: str = "uro"
    base_url: str = "https://www.uroveneer.com/api/product.php"
    request_timeout: int = 30
    delay_between_images: float = 0.05  # Reduced since we're using parallel downloads
    delay_between_sections: float = 0.5
    max_retries: int = 3
    retry_delay: float = 1.0
    page_size: int = 2000
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    # Parallel download settings
    max_workers: int = 4  # Respectful number of concurrent downloads
    batch_size: int = 20  # Process images in batches
    batch_delay: float = 2.0  # Delay between batches to be respectful

class UroveneerScraper:
    """Uroveneer.com image scraper using their API"""
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.user_agent,
            'Referer': 'https://www.uroveneer.com/'
        })
        self.total_downloaded = 0
        self.total_skipped = 0
        self.total_failed = 0
        # Thread-safe counters for parallel downloads
        self._lock = threading.Lock()
        self._download_counter = 0
        self._skip_counter = 0
        self._fail_counter = 0
        
    def sanitize_section_name(self, name: str) -> str:
        """Convert section name to lowercase and remove special characters"""
        name = name.lower().replace(' ', '_')
        return re.sub(r'[^a-z0-9_]', '', name)
    
    def convert_webp_to_png(self, webp_data: bytes) -> Optional[bytes]:
        """Convert WebP image data to PNG format"""
        try:
            image = Image.open(BytesIO(webp_data))
            png_buffer = BytesIO()
            image.save(png_buffer, format='PNG')
            return png_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error converting image: {str(e)}")
            return None
    
    def download_image(self, url: str, filepath: str) -> bool:
        """Download an image from URL and save as PNG with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, timeout=self.config.request_timeout)
                response.raise_for_status()
                
                # Convert WebP to PNG
                png_data = self.convert_webp_to_png(response.content)
                if png_data:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, 'wb') as f:
                        f.write(png_data)
                    return True
                else:
                    logger.error(f"Failed to convert image from {url}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"Failed to download {url} after {self.config.max_retries} attempts")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {str(e)}")
                return False
        
        return False
    
    @dataclass
    class DownloadTask:
        """Represents a single image download task"""
        url: str
        filepath: str
        filename: str
        product_idx: int
        total_products: int
    
    def download_single_image(self, task: 'UroveneerScraper.DownloadTask') -> Tuple[bool, str]:
        """Download a single image (for use in parallel processing)"""
        # Skip if file already exists
        if os.path.exists(task.filepath):
            with self._lock:
                self._skip_counter += 1
            return True, f"Skipped {task.filename} (already exists)"
        
        # Download and convert image
        success = self.download_image(task.url, task.filepath)
        
        with self._lock:
            if success:
                self._download_counter += 1
                return True, f"Downloaded: {task.filename} [{task.product_idx}/{task.total_products}]"
            else:
                self._fail_counter += 1
                return False, f"Failed to download: {task.filename}"
    
    def download_images_parallel(self, tasks: List['UroveneerScraper.DownloadTask'], section_name: str) -> Tuple[int, int, int]:
        """Download images in parallel with rate limiting"""
        total_tasks = len(tasks)
        if total_tasks == 0:
            return 0, 0, 0
        
        logger.info(f"Starting parallel download of {total_tasks} images for section '{section_name}'")
        
        # Reset thread-safe counters
        with self._lock:
            self._download_counter = 0
            self._skip_counter = 0
            self._fail_counter = 0
        
        # Process in batches to be respectful to the server
        for batch_start in range(0, total_tasks, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, total_tasks)
            batch_tasks = tasks[batch_start:batch_end]
            batch_num = (batch_start // self.config.batch_size) + 1
            total_batches = (total_tasks + self.config.batch_size - 1) // self.config.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_tasks)} images)")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks in the batch
                future_to_task = {
                    executor.submit(self.download_single_image, task): task 
                    for task in batch_tasks
                }
                
                # Process completed downloads
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        success, message = future.result()
                        if success and "Downloaded:" in message:
                            logger.info(message)
                        elif not success:
                            logger.error(message)
                        # Skip messages for already existing files are too verbose
                    except Exception as e:
                        logger.error(f"Error downloading {task.filename}: {str(e)}")
                        with self._lock:
                            self._fail_counter += 1
            
            # Add delay between batches to be respectful
            if batch_end < total_tasks:
                logger.info(f"Waiting {self.config.batch_delay}s before next batch...")
                time.sleep(self.config.batch_delay)
        
        # Return final counts
        with self._lock:
            return self._download_counter, self._skip_counter, self._fail_counter
    
    def make_api_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    self.config.base_url, 
                    params=params, 
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"API request failed after {self.config.max_retries} attempts")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error in API request: {str(e)}")
                return None
        
        return None
    
    def get_sections(self) -> List[Dict]:
        """Get all product sections from the API"""
        logger.info("Fetching product sections...")
        
        params = {
            'operation': 'getAliasProducts',
            'page': '0',
            'alias_ids[]': '0',
            'page_size': '24',
            'description_size': '2',
            'special_page': 'products',
            'need_boards': '1',
            'is_products_page': 'all'
        }
        
        data = self.make_api_request(params)
        if not data:
            logger.error("Failed to fetch sections")
            return []
        
        # Debug: Log the actual response structure
        logger.info(f"DEBUG: API response type: {type(data)}")
        if isinstance(data, list) and len(data) > 0:
            logger.info(f"DEBUG: First element type: {type(data[0])}")
            logger.info(f"DEBUG: First element keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
        elif isinstance(data, dict):
            logger.info(f"DEBUG: Response keys: {list(data.keys())}")
        
        # Handle different response structures
        sections = []
        if isinstance(data, list):
            sections = data
        elif isinstance(data, dict):
            sections = data.get('sections', data.get('data', []))
        
        # If still no sections found, the data might be the sections directly
        if not sections and isinstance(data, list):
            sections = data
        elif not sections and isinstance(data, dict):
            # Check if this response contains the sections data directly
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    # Check if this looks like sections data
                    if isinstance(value[0], dict) and 'alias_id' in value[0]:
                        logger.info(f"DEBUG: Found sections in key '{key}'")
                        sections = value
                        break
        
        logger.info(f"Found {len(sections)} sections")
        return sections
    
    def get_products_in_section(self, alias_id: str, total_products: int) -> List[Dict]:
        """Get all products in a specific section"""
        page_size = max(self.config.page_size, total_products + 100)  # Add buffer
        
        params = {
            'operation': 'getAliasProducts',
            'page': '0',
            'alias_ids[]': str(alias_id),
            'page_size': str(page_size),
            'description_size': '2',
            'special_page': 'products',
            'need_boards': '1',
            'is_products_page': 'all'
        }
        
        products_data = self.make_api_request(params)
        if not products_data:
            return []
        
        # Extract products from the specific section response
        if isinstance(products_data, dict) and 'aliases' in products_data:
            aliases = products_data['aliases']
            if isinstance(aliases, list) and len(aliases) > 0:
                # Find the matching section
                for section in aliases:
                    if isinstance(section, dict) and section.get('alias_id') == alias_id:
                        return section.get('products', [])
        
        return []
    
    def process_section(self, section: Dict) -> Tuple[int, int, int]:
        """Process a single section and return (downloaded, skipped, failed) counts"""
        # Validate section data
        if not isinstance(section, dict):
            logger.warning(f"Section is not a dict: {type(section)} - {section}")
            return 0, 0, 1
        
        alias_id = section.get('alias_id')
        alias_name = section.get('alias_name', '')
        total_products = section.get('total_products', 0)
        
        if not alias_id or not alias_name:
            logger.warning(f"Skipping section with missing data: {section}")
            return 0, 0, 1
        
        if total_products == 0:
            logger.info(f"Skipping section '{alias_name}' - no products")
            return 0, 1, 0
        
        section_name = self.sanitize_section_name(alias_name)
        logger.info(f"Processing section: '{alias_name}' -> '{section_name}' ({total_products} products)")
        
        # Make dedicated API call to get ALL products for this section
        products = self.get_products_in_section(alias_id, total_products)
        
        if not products:
            logger.warning(f"No products found for section '{alias_name}'")
            return 0, 0, 1
        
        logger.info(f"Retrieved {len(products)} products from dedicated API call")
        
        # Prepare all download tasks
        download_tasks = []
        validation_failed = 0
        
        for product_idx, product in enumerate(products, 1):
            if not isinstance(product, dict):
                logger.warning(f"Product is not a dict: {type(product)}")
                validation_failed += 1
                continue
            
            product_id = product.get('product_id')
            product_name = product.get('product_name', f'product_{product_id}')
            images = product.get('images', [])
            
            if not product_id:
                validation_failed += 1
                continue
            
            if not images:
                logger.debug(f"Skipping product {product_id} - no images")
                continue
            
            # Create download tasks for all images for this product
            for img_index, image_data in enumerate(images):
                if not isinstance(image_data, dict):
                    logger.warning(f"Image data is not a dict: {type(image_data)}")
                    validation_failed += 1
                    continue
                
                image_url = image_data.get('image_url', '')
                if not image_url:
                    validation_failed += 1
                    continue
                
                # Create filename
                if len(images) == 1:
                    filename = f"{section_name}-{product_id}.png"
                else:
                    filename = f"{section_name}-{product_id}-{img_index + 1}.png"
                
                filepath = os.path.join(self.config.output_folder, filename)
                
                # Create download task
                task = self.DownloadTask(
                    url=image_url,
                    filepath=filepath,
                    filename=filename,
                    product_idx=product_idx,
                    total_products=len(products)
                )
                download_tasks.append(task)
        
        logger.info(f"Created {len(download_tasks)} download tasks for section '{alias_name}'")
        
        # Download all images in parallel
        if download_tasks:
            section_downloaded, section_skipped, section_failed = self.download_images_parallel(
                download_tasks, alias_name
            )
            # Add validation failures to the failed count
            section_failed += validation_failed
        else:
            section_downloaded = 0
            section_skipped = 0
            section_failed = validation_failed
        
        logger.info(f"Section '{alias_name}' complete: {section_downloaded} downloaded, {section_skipped} skipped, {section_failed} failed")
        return section_downloaded, section_skipped, section_failed
    
    def scrape_all(self) -> Dict[str, int]:
        """Main scraping function"""
        logger.info("Starting Uroveneer scraper...")
        
        # Create output folder
        os.makedirs(self.config.output_folder, exist_ok=True)
        
        # Get all sections
        sections = self.get_sections()
        if not sections:
            logger.error("No sections found. Exiting.")
            return {"downloaded": 0, "skipped": 0, "failed": 0}
        
        # Process each section
        for section_idx, section in enumerate(sections, 1):
            logger.info(f"\n--- Processing section {section_idx}/{len(sections)} ---")
            
            downloaded, skipped, failed = self.process_section(section)
            
            self.total_downloaded += downloaded
            self.total_skipped += skipped
            self.total_failed += failed
            
            # Delay between sections
            if section_idx < len(sections):
                time.sleep(self.config.delay_between_sections)
        
        # Final summary
        logger.info(f"\n=== SCRAPING COMPLETE ===")
        logger.info(f"Total images downloaded: {self.total_downloaded}")
        logger.info(f"Total images skipped: {self.total_skipped}")
        logger.info(f"Total failures: {self.total_failed}")
        logger.info(f"Images saved to '{self.config.output_folder}' folder")
        
        # Verify output
        if os.path.exists(self.config.output_folder):
            actual_files = len([f for f in os.listdir(self.config.output_folder) if f.endswith('.png')])
            logger.info(f"Actual files in output folder: {actual_files}")
        
        return {
            "downloaded": self.total_downloaded,
            "skipped": self.total_skipped,
            "failed": self.total_failed
        }

def main():
    """Main function"""
    # Check for required packages
    try:
        import requests
        from PIL import Image
    except ImportError:
        print("Missing required packages. Please install:")
        print("pip install requests pillow")
        return 1
    
    # Create scraper with default config
    config = ScrapingConfig()
    logger.info(f"Parallel download configuration:")
    logger.info(f"  - Max workers: {config.max_workers}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Batch delay: {config.batch_delay}s")
    scraper = UroveneerScraper(config)
    
    # Run scraper
    try:
        results = scraper.scrape_all()
        return 0 if results["failed"] == 0 else 1
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 