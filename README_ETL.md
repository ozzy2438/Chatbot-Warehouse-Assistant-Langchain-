# 🏭 Automated Factory ETL System

**Production-Ready E-Commerce Intelligence Pipeline**

Complete automation system that scrapes Amazon products weekly, transforms them into analytics-ready data, and keeps your voice assistant up-to-date.

---

## 🎯 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   WEEKLY AUTOMATION TRIGGER                 │
│                    (Every Monday 2:00 AM)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   1️⃣ EXTRACT (Scrape) │
            │   Amazon Bestsellers   │
            │   Target: 2000 products│
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   2️⃣ TRANSFORM        │
            │   • Clean data        │
            │   • Add inventory     │
            │   • Find alternatives │
            │   • Quality checks    │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   3️⃣ LOAD             │
            │   • Update vector DB  │
            │   • Create categories │
            │   • Generate dashboard│
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   ✅ READY FOR USE    │
            │   Voice Assistant     │
            └───────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure (Optional)

Edit `etl_config.yaml` to customize:
- Schedule timing
- Scraping targets
- Warehouse locations
- Email notifications

### 3. Run Options

#### Option A: Run Once (Testing)

```bash
# Production run (2000 products)
python etl_orchestrator.py

# Test run (200 products)
python etl_orchestrator.py --test
```

#### Option B: Schedule Weekly Automation

```bash
# Starts scheduler (runs every Monday 2 AM)
python etl_orchestrator.py --schedule
```

Keep this running 24/7 for automatic weekly updates!

---

## 📁 Project Structure

```
Chatbot-Langchain-ProductFinder/
│
├── etl_orchestrator.py          # Main orchestration engine
├── etl_config.yaml              # Configuration file
├── data_quality.py              # Validation & quality checks
├── monitoring_dashboard.py      # Metrics & monitoring
│
├── scrape_bestsellers_2000.py   # Amazon scraper
├── factory_etl_automation.py    # Legacy file watcher (optional)
│
├── chatbot.py                   # AI chatbot (uses processed data)
├── voice_assistant.py           # Voice interface
│
├── data/
│   ├── raw/                     # Raw scraped data
│   ├── processed/               # Categorized outputs
│   ├── archive/                 # Old files (auto-archived)
│   ├── final_product_database.csv
│   ├── products_for_ai.csv
│   ├── products_with_alternatives.csv
│   └── dashboard_summary.csv
│
├── chroma_db/                   # Vector database (auto-updated)
└── etl_metrics.json             # Pipeline metrics history
```

---

## ⚙️ Configuration Guide

### Schedule Configuration

Edit `etl_config.yaml`:

```yaml
scheduling:
  day_of_week: "mon"  # mon, tue, wed, thu, fri, sat, sun
  hour: 2             # 0-23 (2 AM)
  minute: 0
  timezone: "Australia/Melbourne"
```

### Email Alerts

Set environment variables:

```bash
export ETL_EMAIL_FROM="your-email@gmail.com"
export ETL_EMAIL_PASSWORD="your-app-password"
export ETL_EMAIL_TO="admin@company.com,team@company.com"
```

Enable in `etl_config.yaml`:

```yaml
email:
  enabled: true
  notify_on_success: true
  notify_on_failure: true
```

### Scraper Settings

```yaml
scraper:
  target_products: 2000
  max_pages_per_category: 2
  delay_between_pages: [3, 6]  # Random delay to avoid detection
```

---

## 📊 Monitoring Dashboard

View real-time pipeline status:

```bash
python monitoring_dashboard.py
```

**Dashboard shows:**
- Current pipeline status
- Historical run data
- Data quality trends
- Success rates
- Performance metrics

---

## 🔄 ETL Pipeline Details

### EXTRACT Phase

**What it does:**
- Scrapes Amazon bestseller lists
- 12 categories (Electronics, Books, etc.)
- Collects: prices, ratings, reviews, ranks

**Output:**
- `data/raw/raw_products_TIMESTAMP.csv`

### TRANSFORM Phase

**What it does:**

1. **Data Cleaning**
   - Parse prices, ratings, review counts
   - Handle missing values
   - Type conversions

2. **Data Enrichment**
   - Add warehouse locations (Melbourne, Sydney, Brisbane, Perth)
   - Generate stock quantities (50-500 units)
   - Calculate restock dates
   - Add supplier lead times

3. **Alternative Products**
   - Find similar-priced alternatives (±30%)
   - Find higher-rated alternatives (+0.3 rating)

4. **Quality Validation**
   - Check required fields
   - Validate ranges (price: $0.01-$10,000, rating: 0-5)
   - Remove duplicates
   - Calculate quality score (0-100%)

5. **Categorization**
   - **Low Stock**: Below minimum threshold
   - **High Performers**: Rating ≥4.6 + Sales >30/day
   - **Restock Queue**: Restock within 7 days

**Outputs:**
- `final_product_database.csv` - Main database
- `products_for_ai.csv` - AI-ready descriptions
- `products_with_alternatives.csv` - With recommendations
- `dashboard_summary.csv` - Executive summary
- `processed/low_stock_alerts_TIMESTAMP.csv`
- `processed/high_performers_TIMESTAMP.csv`
- `processed/restock_queue_TIMESTAMP.csv`

### LOAD Phase

**What it does:**
- Rebuilds ChromaDB vector database
- Creates embeddings for semantic search
- Archives old files (>7 days)
- Updates voice assistant data

**Output:**
- Updated `chroma_db/` directory
- Ready for voice assistant queries

---

## 📈 Data Quality

The pipeline includes comprehensive quality checks:

### Validation Rules

| Field | Validation |
|-------|------------|
| product_id | Required, unique |
| product_name | Required, non-empty |
| price | Numeric, $0.01 - $10,000 |
| rating | Numeric, 0.0 - 5.0 |
| stock_quantity | Integer, 0 - 10,000 |

### Quality Score

Pipeline calculates a quality score (0-100%):
- **90-100%**: Excellent - No issues
- **80-89%**: Good - Minor warnings
- **70-79%**: Fair - Some issues
- **<70%**: Poor - Review required

---

## 🎛️ Advanced Usage

### Custom Schedule (Cron-style)

```python
from etl_orchestrator import ETLScheduler

scheduler = ETLScheduler()
scheduler.scheduler.add_job(
    scheduler.run_pipeline,
    CronTrigger(day_of_week='wed', hour=3, minute=30),
    id='custom_schedule'
)
scheduler.scheduler.start()
```

### Manual Pipeline Execution

```python
from etl_orchestrator import ETLPipeline

pipeline = ETLPipeline(test_mode=False)
stats = pipeline.run()

print(f"Extracted: {stats['extract_count']} products")
print(f"Quality Score: {stats['transform']['quality_score']:.2f}%")
```

### Data Quality Testing

```python
from data_quality import DataQualityValidator
import pandas as pd

df = pd.read_csv("data/raw/raw_products_123.csv")
validator = DataQualityValidator()
clean_df, issues = validator.validate(df)

print(validator.get_report())
```

---

## 🔧 Troubleshooting

### Pipeline Fails at Extract

**Issue:** Selenium can't find Chrome driver

**Fix:**
```bash
# Install Chrome if not present
brew install --cask google-chrome  # macOS

# webdriver-manager will auto-download driver
```

**Issue:** Amazon blocks scraping

**Fix:**
- Increase delays in `etl_config.yaml`
- Run in non-headless mode to check CAPTCHA
- Use VPN or rotate IP addresses

### Pipeline Fails at Transform

**Issue:** Quality score too low

**Fix:**
- Check `data/raw/` for corrupted CSV
- Review validation errors in logs
- Adjust thresholds in `data_quality.py`

### Pipeline Fails at Load

**Issue:** Vector DB creation fails

**Fix:**
```bash
# Clear old vector DB
rm -rf chroma_db/

# Rebuild
python etl_orchestrator.py
```

### Scheduler Not Running

**Issue:** Process stops after SSH disconnect

**Fix:**
```bash
# Run with nohup
nohup python etl_orchestrator.py --schedule > etl.log 2>&1 &

# Or use systemd (Linux)
sudo systemctl enable etl-pipeline.service
```

---

## 📧 Email Notifications

Sample email format:

```
Subject: ✅ SUCCESS: ETL Pipeline 20231201_020000

ETL Pipeline Execution Report
==============================

Run ID: 20231201_020000
Status: SUCCESS
Duration: 847s

Statistics:
-----------
- Products Extracted: 2000
- Products Transformed: 1998
- Vector Embeddings: 1998
- Quality Score: 96.5%

Categories:
-----------
- Low Stock: 62 products
- High Performers: 147 products
- Restock Queue: 89 products
```

---

## 🎯 Integration with Voice Assistant

After pipeline completes, voice assistant automatically uses new data:

```bash
# Start voice assistant
python voice_assistant.py

# Or text-based chatbot
python chatbot.py
```

**Sample queries:**
- "Kaç tane Fire TV Stick stokta var?"
- "En yüksek rating'li Electronics ürünleri hangileri?"
- "Melbourne deposunda low stock olan ürünler?"
- "Bu ürün yoksa alternatif öner"

---

## 📊 Metrics & Analytics

View pipeline metrics:

```bash
# View metrics file
cat etl_metrics.json

# View dashboard
python monitoring_dashboard.py
```

**Tracked metrics:**
- Run duration
- Products extracted/transformed/loaded
- Quality scores over time
- Success/failure rates
- Error logs

---

## 🚀 Deployment Options

### Option 1: Local Server (Simple)

```bash
# Run scheduler 24/7
nohup python etl_orchestrator.py --schedule > etl.log 2>&1 &

# Monitor
tail -f etl.log
```

### Option 2: Cron Job (Traditional)

```bash
# Add to crontab
crontab -e

# Every Monday at 2 AM
0 2 * * 1 cd /path/to/project && /usr/bin/python3 etl_orchestrator.py
```

### Option 3: Docker (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "etl_orchestrator.py", "--schedule"]
```

```bash
docker build -t etl-pipeline .
docker run -d --name etl -v $(pwd)/data:/app/data etl-pipeline
```

### Option 4: Cloud (AWS/Azure)

Use AWS Lambda + EventBridge or Azure Functions + Timer trigger

---

## 🔒 Security Best Practices

1. **Never commit API keys**
   ```bash
   echo "OPENAI_API_KEY=your-key" >> .env
   echo ".env" >> .gitignore
   ```

2. **Use environment variables**
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ETL_EMAIL_PASSWORD="app-password"
   ```

3. **Restrict file permissions**
   ```bash
   chmod 600 .env
   chmod 700 data/
   ```

4. **Monitor access logs**
   ```bash
   tail -f etl_orchestrator.log
   ```

---

## 📝 Customization Examples

### Change Warehouse Locations

Edit `etl_config.yaml`:

```yaml
transformation:
  warehouses:
    - "New York"
    - "Los Angeles"
    - "Chicago"
    - "Houston"
```

### Add More Categories

Edit `etl_config.yaml`:

```yaml
scraper:
  categories:
    - "Electronics"
    - "Books"
    - "Video Games"  # Add new
    - "Music"        # Add new
```

### Adjust Stock Ranges

Edit `etl_config.yaml`:

```yaml
transformation:
  stock_range: [100, 1000]  # Higher stock
  threshold_range: [20, 80]  # Higher thresholds
```

---

## 🆘 Support & Contact

**Issues?**
- Check logs: `etl_orchestrator.log`
- Review metrics: `etl_metrics.json`
- Test mode: `python etl_orchestrator.py --test`

**Need help?**
- GitHub Issues: [Link to repo]
- Email: your-support@email.com
- Docs: This file!

---

## 📜 License

MIT License - Feel free to use and modify for your projects.

---

## 🎉 What's Next?

- ✅ Automated weekly scraping
- ✅ Data transformation & quality
- ✅ Vector DB updates
- ✅ Voice assistant integration
- 🔄 Add Slack notifications
- 🔄 Add web dashboard (Flask/Streamlit)
- 🔄 Add A/B testing for scraping strategies
- 🔄 Multi-region support (US, UK, EU)

---

**Built with ❤️ for modern e-commerce operations**

