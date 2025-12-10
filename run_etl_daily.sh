#!/bin/bash
# ETL Automation Script
# Runs weekly on Monday at 6 PM via cron

cd /Users/osmanorka/Chatbot-Langchain-ProductFinder
source venv/bin/activate

# Log file
LOG_FILE="logs/etl_automation_$(date +%Y%m%d).log"
mkdir -p logs

echo "========================================" >> $LOG_FILE
echo "ETL Run Started: $(date)" >> $LOG_FILE
echo "========================================" >> $LOG_FILE

# Send start notification (macOS notification)
osascript -e 'display notification "ETL Pipeline başlıyor..." with title "Amazon ETL" sound name "default"'

# Run ETL
python etl_orchestrator.py >> $LOG_FILE 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ETL completed successfully at $(date)" >> $LOG_FILE
    osascript -e 'display notification "ETL başarıyla tamamlandı! ✅" with title "Amazon ETL" sound name "default"'
else
    echo "❌ ETL failed at $(date)" >> $LOG_FILE
    osascript -e 'display notification "ETL başarısız! ❌ Log kontrol et." with title "Amazon ETL" sound name "default"'
fi

echo "" >> $LOG_FILE
