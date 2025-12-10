# Quick Start: Airflow ETL Automation

## ⚡ Simple Setup (Airflow 3.x)

Your Airflow DAG is ready. Just run:

```bash
cd /Users/osmanorka/Chatbot-Langchain-ProductFinder

# Set DAGs folder
export AIRFLOW__CORE__DAGS_FOLDER=/Users/osmanorka/Chatbot-Langchain-ProductFinder/airflow_dags

# Start Airflow (all-in-one: webserver + scheduler + database)
airflow standalone
```

**That's it!** This single command:
- ✅ Initializes database
- ✅ Creates admin user (check terminal for password)
- ✅ Starts webserver on port 8080
- ✅ Starts scheduler
- ✅ Loads your DAG

## 📱 Access Airflow UI

1. Wait ~30 seconds for startup
2. Look in terminal for:
   ```
   standalone | Airflow is ready
   standalone | Login with username: admin  password: <YOUR_PASSWORD>
   ```
3. Open: **http://localhost:8080**
4. Login with credentials from step 2
5. Find DAG: **amazon_etl_pipeline**
6. Toggle it ON (activate)

## 🎯 What Happens

**Automatically:** Runs daily at 2 AM

**Manually:** Click "▶ Trigger DAG" button anytime

**Pipeline Steps:**
1. **Scrape** → 2000 Amazon products
2. **Transform** → Clean + enrich data
3. **Load** → Update vector DB
4. **Cleanup** → Archive old files
5. **Summary** → Print metrics

## Alternative: Run ETL Manually (No Airflow)

If you don't want to use Airflow:

```bash
# Test mode (200 products)
python etl_orchestrator.py --test

# Full run (2000 products)
python etl_orchestrator.py
```

---

## 📝 Notes

- **Airflow logs**: `~/airflow/logs/`
- **Database**: `~/airflow/airflow.db` (SQLite)
- **Stop**: Press `Ctrl+C` in terminal

Tümü hazır! Sadece `airflow standalone` komutunu çalıştır! 🚀
