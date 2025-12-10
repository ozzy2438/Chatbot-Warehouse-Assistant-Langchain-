"""
Airflow DAG: Product Chatbot ETL Pipeline
==========================================
Automated pipeline for scraping, transforming, and updating the chatbot's knowledge base.

Schedule: Daily at 2 AM
Author: AI Operations Assistant
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator

# Project paths
PROJECT_DIR = Path("/Users/osmanorka/Chatbot-Langchain-ProductFinder")
sys.path.insert(0, str(PROJECT_DIR))

# Default args for all tasks
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email': ['alerts@yourcompany.com'],  # Update with your email
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# DAG definition
dag = DAG(
    'product_chatbot_pipeline',
    default_args=default_args,
    description='Scrape Amazon → Transform → Update Chatbot Vector Store',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2025, 12, 1),  # Start date (fixed date instead of days_ago)
    catchup=False,
    tags=['etl', 'chatbot', 'amazon', 'langchain'],
)


def scrape_amazon_products(**context):
    """Task 1: Scrape Amazon bestsellers."""
    from scrape_bestsellers_2000 import main as scrape_main

    print("📦 Starting Amazon scraping...")
    result = scrape_main()

    # Push result to XCom for next tasks
    context['ti'].xcom_push(key='scrape_count', value=result.get('count', 0))

    print(f"✓ Scraped {result.get('count', 0)} products")
    return result


def transform_data(**context):
    """Task 2: Transform and clean data."""
    from etl_orchestrator import ProductETLOrchestrator

    print("🔄 Starting data transformation...")

    orchestrator = ProductETLOrchestrator()
    result = orchestrator.run_pipeline()

    # Push metrics to XCom
    context['ti'].xcom_push(key='transform_metrics', value=result)

    print(f"✓ Transformation complete: {result}")
    return result


def rebuild_vectorstore(**context):
    """Task 3: Rebuild chatbot vector store."""
    from chatbot import rebuild_vectorstore

    print("🗄️ Rebuilding vector store...")

    chatbot = rebuild_vectorstore()

    print("✓ Vector store rebuilt successfully")
    return {'status': 'success'}


def run_quality_checks(**context):
    """Task 4: Run data quality checks."""
    from data_quality import DataQualityChecker

    print("✅ Running quality checks...")

    checker = DataQualityChecker(PROJECT_DIR / "data" / "products_for_ai.csv")
    report = checker.run_all_checks()

    # Push report to XCom
    context['ti'].xcom_push(key='quality_report', value=report)

    # Fail if critical issues found
    if report.get('critical_issues', 0) > 0:
        raise ValueError(f"Critical data quality issues found: {report}")

    print(f"✓ Quality checks passed: {report}")
    return report


def send_completion_alert(**context):
    """Task 5: Send completion notification."""
    ti = context['ti']

    # Pull metrics from previous tasks
    scrape_count = ti.xcom_pull(key='scrape_count', task_ids='scrape_amazon')
    transform_metrics = ti.xcom_pull(key='transform_metrics', task_ids='transform_data')
    quality_report = ti.xcom_pull(key='quality_report', task_ids='quality_checks')

    message = f"""
    ✅ Product Chatbot Pipeline Completed
    =====================================

    📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    📊 Metrics:
    - Products Scraped: {scrape_count}
    - Transform Status: {transform_metrics.get('status', 'N/A')}
    - Quality Score: {quality_report.get('quality_score', 'N/A')}%

    🤖 Chatbot vector store updated and ready!
    """

    print(message)

    # TODO: Send email/Slack notification
    # For now, just log

    return {'status': 'notification_sent'}


# Define tasks
task_scrape = PythonOperator(
    task_id='scrape_amazon',
    python_callable=scrape_amazon_products,
    dag=dag,
)

task_transform = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

# Use BashOperator for vector rebuild (more reliable for long-running tasks)
task_rebuild = BashOperator(
    task_id='rebuild_vectorstore',
    bash_command=f'cd {PROJECT_DIR} && source venv/bin/activate && python chatbot.py --rebuild',
    dag=dag,
)

task_quality = PythonOperator(
    task_id='quality_checks',
    python_callable=run_quality_checks,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_completion_alert,
    dag=dag,
)

# Define task dependencies
task_scrape >> task_transform >> task_rebuild >> task_quality >> task_notify
