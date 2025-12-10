"""
ETL ORCHESTRATOR - Automated E-Commerce Intelligence Pipeline
==============================================================
Weekly automated system that:
1. EXTRACT: Scrapes Amazon products
2. TRANSFORM: Cleans, enriches, and categorizes data
3. LOAD: Updates vector DB and prepares for voice assistant

Schedule: Every Monday at 2:00 AM (configurable)
Monitoring: Email alerts, logs, and dashboard updates

Usage:
    python etl_orchestrator.py              # Run once immediately
    python etl_orchestrator.py --schedule   # Run with weekly scheduler
    python etl_orchestrator.py --test       # Test mode (small dataset)
"""

import os
import sys
import json
import time
import logging
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import traceback

# Scheduling
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Email alerts
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Rich output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import our existing modules
from scrape_bestsellers_2000 import main as scrape_main
from chatbot import ProductChatbot, rebuild_vectorstore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# Configuration
class ETLConfig:
    """ETL Configuration"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    ARCHIVE_DIR = DATA_DIR / "archive"
    CHROMA_DIR = PROJECT_ROOT / "chroma_db"
    
    # Scraper settings
    SCRAPE_TARGET = 2000  # Target number of products
    SCRAPE_CATEGORIES = 12  # Number of categories
    
    # Transformation settings
    PRICE_VARIANCE_THRESHOLD = 0.30  # 30% for similar price alternatives
    MIN_RATING_DIFF = 0.3  # Minimum rating difference for alternatives
    LOW_STOCK_MULTIPLIER = 0.5  # Below 50% of threshold = low stock
    
    # Scheduling (Monday 2 AM)
    SCHEDULE_DAY = 'mon'
    SCHEDULE_HOUR = 2
    SCHEDULE_MINUTE = 0
    
    # Email alerts (optional)
    ENABLE_EMAIL = False  # Set True and configure SMTP
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_FROM = os.getenv("ETL_EMAIL_FROM", "")
    EMAIL_PASSWORD = os.getenv("ETL_EMAIL_PASSWORD", "")
    EMAIL_TO = os.getenv("ETL_EMAIL_TO", "").split(",")
    
    # Monitoring
    MAX_RETRIES = 3
    RETRY_DELAY = 300  # 5 minutes
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.RAW_DIR, cls.PROCESSED_DIR, cls.ARCHIVE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


class ETLPipeline:
    """Main ETL Pipeline Orchestrator"""
    
    def __init__(self, test_mode: bool = False):
        self.console = Console()
        self.test_mode = test_mode
        self.config = ETLConfig()
        self.config.setup_directories()
        
        # Pipeline state
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats = {
            'start_time': None,
            'end_time': None,
            'extract_count': 0,
            'transform_count': 0,
            'load_count': 0,
            'errors': [],
            'status': 'pending'
        }
    
    def run(self) -> Dict:
        """Execute full ETL pipeline"""
        self.stats['start_time'] = datetime.now()
        
        try:
            self.console.print()
            self.console.print(Panel.fit(
                f"[bold green]🚀 ETL Pipeline Starting[/bold green]\n"
                f"[cyan]Run ID: {self.run_id}[/cyan]\n"
                f"[dim]Mode: {'TEST' if self.test_mode else 'PRODUCTION'}[/dim]",
                border_style="green"
            ))
            
            # Step 1: Extract
            extract_result = self._extract()
            if not extract_result['success']:
                raise Exception(f"Extract failed: {extract_result['error']}")
            
            # Step 2: Transform
            transform_result = self._transform(extract_result['file_path'])
            if not transform_result['success']:
                raise Exception(f"Transform failed: {transform_result['error']}")
            
            # Step 3: Load
            load_result = self._load(transform_result['files'])
            if not load_result['success']:
                raise Exception(f"Load failed: {load_result['error']}")
            
            # Success!
            self.stats['status'] = 'success'
            self.stats['end_time'] = datetime.now()
            
            self._print_summary()
            self._send_notification(success=True)
            
            return self.stats
            
        except Exception as e:
            self.stats['status'] = 'failed'
            self.stats['end_time'] = datetime.now()
            self.stats['errors'].append(str(e))
            
            logger.error(f"ETL Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            
            self.console.print(f"\n[bold red]❌ Pipeline Failed: {e}[/bold red]")
            self._send_notification(success=False, error=str(e))
            
            raise
    
    def _extract(self) -> Dict:
        """Step 1: Extract data from Amazon"""
        self.console.print("\n[bold cyan]📥 STEP 1: EXTRACT[/bold cyan]")
        self.console.print("Scraping Amazon bestsellers...")
        
        try:
            # Modify target for test mode
            if self.test_mode:
                os.environ['SCRAPE_TARGET'] = '200'
                self.console.print("[yellow]Test Mode: Scraping 200 products[/yellow]")
            
            # Run scraper
            logger.info("Starting Amazon scraper...")
            scrape_main()
            
            # Find generated CSV
            csv_file = self.config.PROJECT_ROOT / "amazon_bestsellers_2000.csv"
            if not csv_file.exists():
                raise FileNotFoundError("Scraper did not generate CSV file")
            
            # Move to raw directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_file = self.config.RAW_DIR / f"raw_products_{timestamp}.csv"
            shutil.move(str(csv_file), str(raw_file))
            
            # Count products
            df = pd.read_csv(raw_file)
            self.stats['extract_count'] = len(df)
            
            self.console.print(f"[green]✓ Extracted {len(df)} products[/green]")
            logger.info(f"Extract complete: {len(df)} products")
            
            return {
                'success': True,
                'file_path': raw_file,
                'count': len(df)
            }
            
        except Exception as e:
            logger.error(f"Extract failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _transform(self, raw_file: Path) -> Dict:
        """Step 2: Transform and enrich data"""
        self.console.print("\n[bold cyan]🔄 STEP 2: TRANSFORM[/bold cyan]")
        self.console.print("Cleaning, enriching, and categorizing data...")
        
        try:
            # Load raw data
            df = pd.read_csv(raw_file)
            logger.info(f"Loaded {len(df)} raw products")
            
            # Clean and transform
            df_clean = self._clean_data(df)
            df_enriched = self._enrich_data(df_clean)
            df_final = self._add_alternatives(df_enriched)
            
            # Generate AI descriptions
            df_ai = self._generate_ai_descriptions(df_final)
            
            # Create category splits
            categories = self._split_categories(df_final)
            
            # Save all outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            files = {
                'final': self.config.DATA_DIR / "final_product_database.csv",
                'ai': self.config.DATA_DIR / "products_for_ai.csv",
                'alternatives': self.config.DATA_DIR / "products_with_alternatives.csv",
                'low_stock': self.config.PROCESSED_DIR / f"low_stock_alerts_{timestamp}.csv",
                'high_performers': self.config.PROCESSED_DIR / f"high_performers_{timestamp}.csv",
                'restock_queue': self.config.PROCESSED_DIR / f"restock_queue_{timestamp}.csv",
            }
            
            # Save main files
            df_final.to_csv(files['final'], index=False)
            df_ai.to_csv(files['ai'], index=False)
            df_final.to_csv(files['alternatives'], index=False)
            
            # Save categories
            categories['low_stock'].to_csv(files['low_stock'], index=False)
            categories['high_performers'].to_csv(files['high_performers'], index=False)
            categories['restock_queue'].to_csv(files['restock_queue'], index=False)
            
            # Generate dashboard
            self._generate_dashboard(df_final)
            
            self.stats['transform_count'] = len(df_final)
            
            self.console.print(f"[green]✓ Transformed {len(df_final)} products[/green]")
            self.console.print(f"  • Low Stock: {len(categories['low_stock'])} products")
            self.console.print(f"  • High Performers: {len(categories['high_performers'])} products")
            self.console.print(f"  • Restock Queue: {len(categories['restock_queue'])} products")
            
            logger.info(f"Transform complete: {len(df_final)} products")
            
            return {
                'success': True,
                'files': files,
                'count': len(df_final),
                'categories': categories
            }
            
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        self.console.print("  → Cleaning data...")
        
        df_clean = df.copy()
        
        # Clean price
        if 'discounted_price' in df_clean.columns:
            df_clean['price'] = df_clean['discounted_price'].str.replace('$', '').str.replace(',', '')
            df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')
        
        # Clean rating
        if 'rating' in df_clean.columns:
            df_clean['rating'] = df_clean['rating'].str.extract(r'([\d.]+)')[0]
            df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
        
        # Clean review count
        if 'review_count' in df_clean.columns:
            df_clean['review_count'] = df_clean['review_count'].str.replace(',', '')
            df_clean['review_count'] = pd.to_numeric(df_clean['review_count'], errors='coerce')
        
        # Rename columns
        column_mapping = {
            'product_id': 'asin',
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Fill missing values
        df_clean['price'] = df_clean['price'].fillna(df_clean['price'].median())
        df_clean['rating'] = df_clean['rating'].fillna(4.5)
        df_clean['review_count'] = df_clean['review_count'].fillna(0)
        
        return df_clean
    
    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add inventory and operational fields"""
        self.console.print("  → Enriching with inventory data...")
        
        import numpy as np
        
        df_enriched = df.copy()
        
        # Add warehouse locations (round-robin)
        warehouses = ['Melbourne', 'Sydney', 'Brisbane', 'Perth']
        df_enriched['warehouse_location'] = [warehouses[i % len(warehouses)] for i in range(len(df_enriched))]
        
        # Add stock quantities (simulated based on rating and category)
        base_stock = np.random.randint(50, 500, len(df_enriched))
        df_enriched['stock_quantity'] = base_stock
        
        # Add minimum stock threshold
        df_enriched['minimum_stock_threshold'] = np.random.randint(10, 50, len(df_enriched))
        
        # Add restock dates (7-30 days in future)
        days_ahead = np.random.randint(7, 30, len(df_enriched))
        df_enriched['restock_date'] = [
            (datetime.now() + timedelta(days=int(d))).strftime('%Y-%m-%d')
            for d in days_ahead
        ]
        
        # Add supplier lead time
        df_enriched['supplier_lead_time_days'] = np.random.randint(3, 14, len(df_enriched))
        
        # Add last sold date (recent)
        days_ago = np.random.randint(0, 30, len(df_enriched))
        df_enriched['last_sold_date'] = [
            (datetime.now() - timedelta(days=int(d))).strftime('%Y-%m-%d')
            for d in days_ago
        ]
        
        # Add daily sales average
        df_enriched['daily_sales_avg'] = np.random.randint(1, 50, len(df_enriched))
        
        # Add satisfaction metrics
        df_enriched['sentiment_score'] = (df_enriched['rating'] * 20).round(1)
        df_enriched['satisfaction_level'] = pd.cut(
            df_enriched['rating'],
            bins=[0, 4.0, 4.3, 4.6, 5.1],
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )
        df_enriched['recommendation_score'] = ((df_enriched['rating'] / 5) * 100).round(1)
        df_enriched['complaint_risk'] = df_enriched['rating'].apply(
            lambda x: 'High' if x < 4.0 else 'Low'
        )
        
        return df_enriched
    
    def _add_alternatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find similar and better alternatives for each product"""
        self.console.print("  → Computing product alternatives...")
        
        df_alt = df.copy()
        
        # Initialize columns
        df_alt['similar_price_alternatives'] = ''
        df_alt['higher_rated_alternatives'] = ''
        
        for idx, row in df_alt.iterrows():
            # Similar price alternatives (within 30%)
            price_min = row['price'] * 0.7
            price_max = row['price'] * 1.3
            similar = df_alt[
                (df_alt['category'] == row['category']) &
                (df_alt['price'] >= price_min) &
                (df_alt['price'] <= price_max) &
                (df_alt['asin'] != row['asin'])
            ].head(5)['asin'].tolist()
            
            df_alt.at[idx, 'similar_price_alternatives'] = '|'.join(similar) if similar else ''
            
            # Higher rated alternatives
            higher_rated = df_alt[
                (df_alt['category'] == row['category']) &
                (df_alt['rating'] > row['rating'] + 0.3) &
                (df_alt['asin'] != row['asin'])
            ].nlargest(5, 'rating')['asin'].tolist()
            
            df_alt.at[idx, 'higher_rated_alternatives'] = '|'.join(higher_rated) if higher_rated else ''
        
        return df_alt
    
    def _generate_ai_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate natural language descriptions"""
        self.console.print("  → Generating AI descriptions...")
        
        df_ai = df.copy()
        
        descriptions = []
        for _, row in df_ai.iterrows():
            desc = f"{row['product_name']} in {row['category']} category. "
            desc += f"Priced at ${row['price']:.2f} with {row['rating']}/5 rating. "
            desc += f"Stock: {row['stock_quantity']} units at {row['warehouse_location']}. "
            
            if row['stock_quantity'] < row['minimum_stock_threshold']:
                desc += f"⚠️ LOW STOCK - Restock scheduled for {row['restock_date']}. "
            
            descriptions.append(desc)
        
        df_ai['product_description'] = descriptions
        
        return df_ai
    
    def _split_categories(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into operational categories"""
        categories = {}
        
        # Low stock products
        categories['low_stock'] = df[
            df['stock_quantity'] < df['minimum_stock_threshold']
        ].copy()
        
        # High performers
        categories['high_performers'] = df[
            (df['rating'] >= 4.6) & (df['daily_sales_avg'] > 30)
        ].copy()
        
        # Restock queue (7 days)
        df['days_until_restock'] = pd.to_datetime(df['restock_date']).apply(
            lambda x: (x - datetime.now()).days
        )
        categories['restock_queue'] = df[df['days_until_restock'] <= 7].copy()
        
        return categories
    
    def _generate_dashboard(self, df: pd.DataFrame):
        """Generate executive dashboard summary"""
        self.console.print("  → Generating dashboard...")
        
        # Category summary
        summary = df.groupby('category').agg({
            'asin': 'count',
            'price': 'mean',
            'rating': 'mean',
            'stock_quantity': 'sum',
            'daily_sales_avg': 'sum'
        }).round(2)
        
        summary.columns = ['total_products', 'avg_price', 'avg_rating', 'total_stock', 'daily_sales']
        
        # Warehouse summary
        warehouse = df.groupby('warehouse_location').agg({
            'asin': 'count',
            'stock_quantity': 'sum'
        })
        
        # Save
        summary.to_csv(self.config.DATA_DIR / "dashboard_summary.csv")
        warehouse.to_csv(self.config.DATA_DIR / "warehouse_summary.csv")
    
    def _load(self, files: Dict) -> Dict:
        """Step 3: Load into vector DB and prepare for assistant"""
        self.console.print("\n[bold cyan]📤 STEP 3: LOAD[/bold cyan]")
        self.console.print("Updating vector database...")
        
        try:
            # Rebuild ChromaDB vector store
            logger.info("Rebuilding vector store...")
            
            # Remove old vector store
            if self.config.CHROMA_DIR.exists():
                shutil.rmtree(self.config.CHROMA_DIR)
                logger.info("Old vector store removed")
            
            # Initialize new chatbot (will create vector store)
            chatbot = ProductChatbot()
            
            # Count embeddings
            collection = chatbot.vectorstore._collection
            self.stats['load_count'] = collection.count()
            
            self.console.print(f"[green]✓ Vector DB updated with {self.stats['load_count']} embeddings[/green]")
            logger.info(f"Load complete: {self.stats['load_count']} embeddings")
            
            # Archive old raw files
            self._archive_old_files()
            
            return {
                'success': True,
                'count': self.stats['load_count']
            }
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _archive_old_files(self):
        """Archive old raw and processed files"""
        self.console.print("  → Archiving old files...")
        
        # Move raw files older than 7 days
        cutoff = datetime.now() - timedelta(days=7)
        
        for raw_file in self.config.RAW_DIR.glob("*.csv"):
            if datetime.fromtimestamp(raw_file.stat().st_mtime) < cutoff:
                archive_path = self.config.ARCHIVE_DIR / raw_file.name
                shutil.move(str(raw_file), str(archive_path))
                logger.info(f"Archived: {raw_file.name}")
    
    def _print_summary(self):
        """Print pipeline summary"""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        table = Table(title="ETL Pipeline Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Run ID", self.run_id)
        table.add_row("Status", "✅ SUCCESS")
        table.add_row("Duration", f"{duration:.0f} seconds")
        table.add_row("Products Extracted", str(self.stats['extract_count']))
        table.add_row("Products Transformed", str(self.stats['transform_count']))
        table.add_row("Vector Embeddings", str(self.stats['load_count']))
        
        self.console.print()
        self.console.print(table)
        self.console.print()
        
        logger.info(f"Pipeline completed successfully in {duration:.0f}s")
    
    def _send_notification(self, success: bool, error: str = None):
        """Send email notification"""
        if not self.config.ENABLE_EMAIL:
            return
        
        if not self.config.EMAIL_FROM or not self.config.EMAIL_TO:
            logger.warning("Email not configured, skipping notification")
            return
        
        try:
            subject = f"{'✅ SUCCESS' if success else '❌ FAILED'}: ETL Pipeline {self.run_id}"
            
            body = f"""
ETL Pipeline Execution Report
==============================

Run ID: {self.run_id}
Status: {'SUCCESS' if success else 'FAILED'}
Duration: {(self.stats['end_time'] - self.stats['start_time']).total_seconds():.0f}s

Statistics:
-----------
- Products Extracted: {self.stats['extract_count']}
- Products Transformed: {self.stats['transform_count']}
- Vector Embeddings: {self.stats['load_count']}

"""
            
            if not success:
                body += f"\nError Details:\n{error}\n"
            
            msg = MIMEMultipart()
            msg['From'] = self.config.EMAIL_FROM
            msg['To'] = ', '.join(self.config.EMAIL_TO)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls()
                server.login(self.config.EMAIL_FROM, self.config.EMAIL_PASSWORD)
                server.send_message(msg)
            
            logger.info("Email notification sent")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")


class ETLScheduler:
    """Scheduled ETL execution"""
    
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.config = ETLConfig()
    
    def schedule_weekly(self):
        """Schedule weekly ETL run"""
        console.print()
        console.print(Panel.fit(
            f"[bold green]⏰ ETL Scheduler Started[/bold green]\n"
            f"[cyan]Schedule: Every {self.config.SCHEDULE_DAY.upper()} at "
            f"{self.config.SCHEDULE_HOUR:02d}:{self.config.SCHEDULE_MINUTE:02d}[/cyan]\n"
            f"[dim]Press Ctrl+C to stop[/dim]",
            border_style="green"
        ))
        
        # Add job
        self.scheduler.add_job(
            self.run_pipeline,
            CronTrigger(
                day_of_week=self.config.SCHEDULE_DAY,
                hour=self.config.SCHEDULE_HOUR,
                minute=self.config.SCHEDULE_MINUTE
            ),
            id='etl_weekly',
            name='Weekly ETL Pipeline',
            max_instances=1
        )
        
        logger.info(f"Scheduled weekly ETL: {self.config.SCHEDULE_DAY} {self.config.SCHEDULE_HOUR:02d}:{self.config.SCHEDULE_MINUTE:02d}")
        
        # Show next run
        next_run = self.scheduler.get_job('etl_weekly').next_run_time
        console.print(f"\n[yellow]Next run: {next_run}[/yellow]\n")
        
        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            console.print("\n[yellow]Scheduler stopped[/yellow]")
            self.scheduler.shutdown()
    
    def run_pipeline(self):
        """Execute pipeline with retry logic"""
        for attempt in range(1, self.config.MAX_RETRIES + 1):
            try:
                logger.info(f"Starting pipeline (attempt {attempt}/{self.config.MAX_RETRIES})")
                
                pipeline = ETLPipeline()
                pipeline.run()
                
                logger.info("Pipeline completed successfully")
                return
                
            except Exception as e:
                logger.error(f"Pipeline attempt {attempt} failed: {e}")
                
                if attempt < self.config.MAX_RETRIES:
                    logger.info(f"Retrying in {self.config.RETRY_DELAY}s...")
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    logger.error("All retry attempts exhausted")
                    raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETL Orchestrator for E-Commerce Intelligence")
    parser.add_argument('--schedule', action='store_true', help='Run with weekly scheduler')
    parser.add_argument('--test', action='store_true', help='Test mode (small dataset)')
    parser.add_argument('--now', action='store_true', help='Run immediately (default)')
    
    args = parser.parse_args()
    
    console = Console()
    console.print()
    console.print("[bold blue]╔══════════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║   ETL Orchestrator - Factory System     ║[/bold blue]")
    console.print("[bold blue]║   Automated E-Commerce Intelligence     ║[/bold blue]")
    console.print("[bold blue]╚══════════════════════════════════════════╝[/bold blue]")
    console.print()
    
    if args.schedule:
        # Run scheduler
        scheduler = ETLScheduler()
        scheduler.schedule_weekly()
    else:
        # Run once immediately
        pipeline = ETLPipeline(test_mode=args.test)
        pipeline.run()


if __name__ == "__main__":
    main()

