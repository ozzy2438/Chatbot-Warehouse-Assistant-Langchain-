# Amazon ETL Automation - Setup Guide

## 📋 System Overview

Your ETL system is now clean and automated with Airflow:

**Pipeline Flow:**
```
1. Scrape → 2. Transform → 3. Load → 4. Cleanup → 5. Summary
```

## 🗂️ Project Structure (Cleaned)

```
/Chatbot-Langchain-ProductFinder/
├── airflow_dags/
│   ├── amazon_etl_pipeline.py     ← NEW: Clean Airflow DAG
│   └── product_chatbot_pipeline.py (old - can remove)
├── data/
│   ├── raw/                        # Scraped CSVs
│   ├── processed/                  # Split categories
│   └── archive/                    # Auto-archived (7+ days)
├── scrape_bestsellers_2000.py     # Selenium scraper
├── etl_orchestrator.py            # ETL transformer (all 6 steps)
├── chatbot.py                     # LLM chatbot
├── app_realtime.py                # Web UI
└── README_ETL.md                  # Original docs
```

## 🚀 How to Use

### Option 1: Run Manually (Test)

```bash
# Test scraper
python scrape_bestsellers_2000.py

# Test full ETL pipeline
python etl_orchestrator.py --test
```

### Option 2: Run with Airflow (Production)

**Setup Airflow (if not already):**
```bash
# Install Airflow (already installed)
# pip install apache-airflow

# Initialize/migrate database
airflow db migrate
```

**Start Airflow (Standalone Mode - Airflow 3.x):**
```bash
# Run all-in-one Airflow (scheduler + webserver)
airflow standalone
```

**Access Airflow:**
1. Wait for startup (creates admin user automatically)
2. Check terminal output for username/password
3. Open: http://localhost:8080
4. Find DAG: `amazon_etl_pipeline`
5. Toggle ON to enable
6. Runs automatically daily at 2 AM

**Manual Trigger:**
Click "Trigger DAG" button to run immediately

## 📊 ETL Pipeline Tasks

### Task 1: Scrape Amazon
- Selenium scrapes 2000 bestsellers
- Output: `data/raw/raw_products_YYYYMMDD_HHMMSS.csv`

### Task 2: Transform Data
- Clean (remove nulls, convert types)
- Enrich (add inventory, customer metrics)
- Split into 4 categories:
  - Low stock alerts
  - High performers
  - Restock queue  
  - Active inventory

### Task 3: Load Vector DB
- Rebuilds ChromaDB vector store
- Creates embeddings for LLM chatbot

### Task 4: Cleanup
- Archives files older than 7 days
- Keeps `data/` directory clean

### Task 5: Summary
- Prints execution metrics
- Shows product counts

## 🔍 Monitoring

**View Logs:**
```bash
# Airflow logs
ls ~/airflow/logs/dag_id=amazon_etl_pipeline/

# ETL orchestrator logs
tail -f etl_orchestrator.log
```

**Check Status:**
- Airflow UI: Task instance details
- XCom values show metrics between tasks

## ⚙️ Configuration

**Schedule:** Daily at 2 AM (change in DAG file)
**Retries:** 3 attempts with 5-minute delay
**Timeout:** 2 hours max execution

**To change schedule:**
Edit `airflow_dags/amazon_etl_pipeline.py`:
```python
schedule='0 2 * * *'  # Cron format
```

## 🧹 Files Removed

**Documentation (13 files):** AIRFLOW_SETUP.md, AUTOMATION.md, etc.
**Old scripts (7 files):** factory_etl_automation.py, monitoring_dashboard.py, etc.
**Old tests:** test_etl_pipeline.py, test_complex.py, test_voice_ui.py

## ✅ System Status

- ✅ Scraper: Working (`scrape_bestsellers_2000.py`)
- ✅ Transformer: All 6 steps in `etl_orchestrator.py`
- ✅ LLM: `chatbot.py` + `app_realtime.py`
- ✅ Airflow: New clean DAG created
- ✅ Project: Cleaned and organized

## 🎯 Next Steps

1. Start Airflow (see setup above)
2. Enable the DAG in UI
3. Trigger manual run to test
4. Let it run daily automatically

Eğer sorun olursa, Airflow UI'dan task logs'una bakabilirsiniz!
