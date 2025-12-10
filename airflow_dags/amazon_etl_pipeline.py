"""
Airflow DAG: Amazon ETL Pipeline (Clean Version)
================================================
Automated daily pipeline: Scrape → Transform → Load

Schedule: Daily at 2:00 AM
Tasks:
  1. Scrape Amazon bestsellers (2000 products)
  2. Transform & clean data (ETL Orchestrator)
  3. Load into Vector DB (ChromaDB)
  4. Archive old files

Author: ETL Automation System
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Project configuration
PROJECT_DIR = Path("/Users/osmanorka/Chatbot-Langchain-ProductFinder")
sys.path.insert(0, str(PROJECT_DIR))

# DAG default arguments
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email_on_failure': False,  # Set to True and add email if needed
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Create DAG
with DAG(
    dag_id='amazon_etl_pipeline',
    default_args=default_args,
    description='Amazon ETL: Scrape → Transform → Load to Vector DB',
    schedule='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2025, 12, 9),
    catchup=False,
    tags=['etl', 'amazon', 'chatbot', 'vectordb'],
) as dag:

    # =====================================================
    # TASK 1: SCRAPE - Extract Amazon bestsellers
    # =====================================================
    def scrape_amazon_task(**context):
        """Scrape 2000 Amazon bestseller products"""
        from scrape_bestsellers_2000 import main as scrape_main
        
        print("🔍 Starting Amazon scraper...")
        result = scrape_main()
        
        # Store count in XCom
        products_count = result.get('count', 0) if isinstance(result, dict) else 0
        context['ti'].xcom_push(key='products_scraped', value=products_count)
        
        print(f"✅ Scraped {products_count} products")
        return {'status': 'success', 'count': products_count}

    task_scrape = PythonOperator(
        task_id='scrape_amazon',
        python_callable=scrape_amazon_task,
    )

    # =====================================================
    # TASK 2: TRANSFORM - Clean, enrich, categorize
    # =====================================================
    def transform_data_task(**context):
        """Run ETL transformation pipeline"""
        from etl_orchestrator import ETLPipeline
        
        print("🔄 Starting ETL transformation...")
        
        # Run pipeline (skip scraping since we already did it)
        pipeline = ETLPipeline(test_mode=False)
        
        # Run only transform and load steps
        # Find the latest raw file
        raw_files = sorted(pipeline.config.RAW_DIR.glob("raw_products_*.csv"))
        if not raw_files:
            raise FileNotFoundError("No raw data found from scraping step")
        
        latest_raw = raw_files[-1]
        print(f"📂 Processing: {latest_raw.name}")
        
        # Transform
        transform_result = pipeline._transform(latest_raw)
        if not transform_result['success']:
            raise Exception(f"Transform failed: {transform_result['error']}")
        
        # Store metrics
        context['ti'].xcom_push(key='products_transformed', value=transform_result['count'])
        
        print(f"✅ Transformed {transform_result['count']} products")
        return transform_result

    task_transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data_task,
    )

    # =====================================================
    # TASK 3: LOAD - Update Vector Database
    # =====================================================
    def load_vectordb_task(**context):
        """Rebuild ChromaDB vector store"""
        from etl_orchestrator import ETLPipeline
        import shutil
        
        print("📤 Updating vector database...")
        
        pipeline = ETLPipeline(test_mode=False)
        
        # Load step
        files = {
            'final': pipeline.config.DATA_DIR / "final_product_database.csv",
            'ai': pipeline.config.DATA_DIR / "products_for_ai.csv",
        }
        
        load_result = pipeline._load(files)
        if not load_result['success']:
            raise Exception(f"Load failed: {load_result['error']}")
        
        # Store metrics
        context['ti'].xcom_push(key='embeddings_created', value=load_result['count'])
        
        print(f"✅ Created {load_result['count']} vector embeddings")
        return load_result

    task_load = PythonOperator(
        task_id='load_vectordb',
        python_callable=load_vectordb_task,
    )

    # =====================================================
    # TASK 4: CLEANUP - Archive old files
    # =====================================================
    def cleanup_task(**context):
        """Archive files older than 7 days"""
        from etl_orchestrator import ETLPipeline
        from datetime import datetime, timedelta
        import shutil
        
        print("🗑️ Cleaning up old files...")
        
        pipeline = ETLPipeline(test_mode=False)
        cutoff = datetime.now() - timedelta(days=7)
        archived_count = 0
        
        # Archive old raw files
        for raw_file in pipeline.config.RAW_DIR.glob("*.csv"):
            file_time = datetime.fromtimestamp(raw_file.stat().st_mtime)
            if file_time < cutoff:
                archive_path = pipeline.config.ARCHIVE_DIR / raw_file.name
                shutil.move(str(raw_file), str(archive_path))
                archived_count += 1
                print(f"  📦 Archived: {raw_file.name}")
        
        print(f"✅ Archived {archived_count} old files")
        return {'archived': archived_count}

    task_cleanup = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=cleanup_task,
    )

    # =====================================================
    # TASK 5: SUMMARY - Print pipeline summary
    # =====================================================
    def summary_task(**context):
        """Print pipeline execution summary"""
        ti = context['ti']
        
        scraped = ti.xcom_pull(key='products_scraped', task_ids='scrape_amazon') or 0
        transformed = ti.xcom_pull(key='products_transformed', task_ids='transform_data') or 0
        embeddings = ti.xcom_pull(key='embeddings_created', task_ids='load_vectordb') or 0
        
        summary = f"""
        ╔═══════════════════════════════════════╗
        ║   ETL PIPELINE COMPLETED ✅           ║
        ╚═══════════════════════════════════════╝
        
        📊 SUMMARY:
        • Products Scraped:     {scraped}
        • Products Transformed: {transformed}
        • Vector Embeddings:    {embeddings}
        
        🕒 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        print(summary)
        
        return {
            'scraped': scraped,
            'transformed': transformed,
            'embeddings': embeddings,
            'status': 'success'
        }

    task_summary = PythonOperator(
        task_id='print_summary',
        python_callable=summary_task,
    )

    # =====================================================
    # TASK DEPENDENCIES (Pipeline Flow)
    # =====================================================
    task_scrape >> task_transform >> task_load >> task_cleanup >> task_summary
