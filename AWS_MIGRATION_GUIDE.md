# 🧭 AWS Migration Guide — Explained Like You're Five

> **A step-by-step, button-by-button guide to moving your Warehouse Assistant project to Amazon Web Services (AWS) and Amazon Bedrock — without breaking anything that already works.**

---

## 📖 How to Read This Guide

This guide assumes you have **never used AWS before**. Every click, every button, every menu item is spelled out. If you follow it top-to-bottom, you will end up with:

- Your project running on a real AWS server that anyone on the internet can visit
- Your OpenAI API key safely hidden (not in a file on your laptop)
- Your data files stored safely in the cloud
- A second "AI brain" from Amazon Bedrock that you can switch to with one setting
- An automatic robot that runs your data pipeline every Monday at 2 AM

**Total time:** About 10 hours, split across 4 phases. You can stop after any phase.
**Total cost:** $0 for the first 12 months (thanks to AWS Free Tier), then less than $1/month.

---

## 🗂️ Table of Contents

1. [Before You Start — Things You Need](#-before-you-start)
2. [Understanding the Big Picture](#-understanding-the-big-picture)
3. [PHASE 1 — Create Your AWS Account](#-phase-1--create-your-aws-account)
4. [PHASE 2 — Foundation (S3 + IAM + Secrets Manager)](#-phase-2--foundation-s3--iam--secrets-manager)
5. [PHASE 3 — Compute (Run the App on EC2)](#-phase-3--compute-run-the-app-on-ec2)
6. [PHASE 4 — Automation (EventBridge + Lambda)](#-phase-4--automation-eventbridge--lambda)
7. [PHASE 5 — Amazon Bedrock (The Second AI Brain)](#-phase-5--amazon-bedrock-the-second-ai-brain)
8. [Code Changes You Will Make](#-code-changes-you-will-make)
9. [Testing Everything Works](#-testing-everything-works)
10. [Troubleshooting](#-troubleshooting)
11. [Cost Calculator](#-cost-calculator)

---

## 🧰 Before You Start

You need these things ready before starting:

| What You Need | How to Get It |
|---|---|
| A credit card | AWS requires one even for free services (they check you're a real person) |
| An email address | Use one you check regularly — AWS sends important emails |
| A phone number | AWS will call/text you to verify identity |
| Your OpenAI API key | The one that is already in your `.env` file |
| Your project folder | The Warehouse Assistant project on your laptop |
| About 2 hours of uninterrupted time | Don't rush Phase 1 — verification can take 10-20 min |

---

## 🗺️ Understanding the Big Picture

Think of AWS like a giant shopping mall. Each "shop" in the mall is called a **service**. You only walk into the shops you need.

Here are the shops we will visit:

| Shop Name | What It Does | Like… |
|---|---|---|
| **IAM** | Controls who is allowed to do what | A security guard with a clipboard |
| **S3** | Stores files in the cloud | A giant digital filing cabinet |
| **Secrets Manager** | Hides passwords and API keys | A safe with a combination lock |
| **EC2** | Gives you a computer in the cloud | Renting a PC that's always on |
| **CloudWatch** | Records what your app is doing | A security camera for your code |
| **Lambda** | Runs small bits of code on demand | A vending machine for code |
| **EventBridge** | Triggers things on a schedule | An alarm clock |
| **Bedrock** | Amazon's AI brain (like OpenAI but Amazon's) | A second opinion doctor |

You don't need to understand them all now. We'll visit each one when we need it.

---

## 🎫 PHASE 1 — Create Your AWS Account

> ⏱️ Time: 20-30 minutes
> 💰 Cost: $0

### Step 1.1 — Go to the AWS signup page

1. Open your web browser (Chrome, Safari, Firefox, whatever you use)
2. Type this into the address bar: **https://aws.amazon.com**
3. Press Enter
4. Look at the **top-right corner** of the page. You'll see an orange button that says **"Create an AWS Account"**
5. Click that orange button

### Step 1.2 — Fill in your email

1. You'll see a box labelled **"Root user email address"**. Type your email here.
2. Below it is a box labelled **"AWS account name"**. Type something like `warehouse-assistant-project` (no spaces).
3. Click the orange **"Verify email address"** button
4. Go check your email. AWS sent you a 6-digit code. Copy it.
5. Paste the code into the box on the AWS page
6. Click **"Verify"**

### Step 1.3 — Create a root password

1. AWS asks you to make a password. Make it strong — at least 12 characters with numbers and symbols.
2. Type it twice (once in each box)
3. Click **"Continue (step 1 of 5)"**

### Step 1.4 — Contact information

1. Select **"Personal"** at the top (unless you have a registered business)
2. Fill in your full name, phone number, country, and address
3. Tick the checkbox that says you agree to the terms
4. Click **"Continue (step 2 of 5)"**

### Step 1.5 — Billing information

1. Enter your credit card details
2. AWS will put a small temporary charge ($1 USD) on the card to verify it — this is refunded
3. Click **"Verify and Continue (step 3 of 5)"**

### Step 1.6 — Identity verification

1. Choose **"Text message (SMS)"** (faster than voice call)
2. Enter your phone number
3. Solve the captcha puzzle
4. Click **"Send SMS"**
5. Check your phone for a 6-digit code
6. Type it in and click **"Continue"**

### Step 1.7 — Choose the support plan

1. AWS tries to sell you a paid support plan — **DON'T BUY IT**
2. Scroll down and select **"Basic support – Free"**
3. Click **"Complete sign up"**

### Step 1.8 — Sign in for the first time

1. Click the button **"Go to the AWS Management Console"**
2. You'll see a sign-in page. Select **"Root user"**
3. Enter your email
4. Enter your password
5. Click **"Sign in"**

✅ **You now have an AWS account!** You should see a page called the **AWS Management Console** — this is your "home base" for everything AWS.

### Step 1.9 — Set your region

Very important! AWS has data centres all over the world. You need to pick one and stick with it.

1. Look at the **top-right corner** of the console
2. You'll see a location name (e.g. "Ohio" or "N. Virginia")
3. Click it. A dropdown appears.
4. Choose the one closest to you:
   - Australia: **Asia Pacific (Sydney) ap-southeast-2**
   - UK/Europe: **Europe (London) eu-west-2**
   - US East Coast: **US East (N. Virginia) us-east-1**
5. Write down which region you picked. You'll need it later.

> ⚠️ **Important:** Once you pick a region, do everything in that same region. If you accidentally switch regions, you won't see your stuff!

---

## 🏗️ PHASE 2 — Foundation (S3 + IAM + Secrets Manager)

> ⏱️ Time: 2 hours
> 💰 Cost: ~$0.40/month (Secrets Manager only)

In this phase we build three things:
1. A **safe** to hide your OpenAI API key (Secrets Manager)
2. A **filing cabinet** to store your data files (S3)
3. A **security guard** to control who can access them (IAM)

---

### 🔐 Step 2.1 — Hide your OpenAI Key in Secrets Manager

#### 2.1.1 — Open Secrets Manager

1. At the **top of the AWS Console**, find the **search bar** (it has a magnifying glass icon)
2. Click the search bar
3. Type **"Secrets Manager"**
4. A dropdown appears. Click the first result: **"Secrets Manager"**

#### 2.1.2 — Create your first secret

1. You'll see a page titled "Secrets Manager". Click the orange button **"Store a new secret"**
2. On the next page, under **"Secret type"**, choose **"Other type of secret"**
3. Scroll down to **"Key/value pairs"**
4. You'll see two boxes side by side:
   - In the **left box** (key), type: `OPENAI_API_KEY`
   - In the **right box** (value), paste your real OpenAI key (starts with `sk-...`)
5. Click **"+ Add row"** to add another
   - Left box: `LLM_PROVIDER`
   - Right box: `openai`
6. Leave encryption as default (aws/secretsmanager)
7. Click the orange **"Next"** button at the bottom

#### 2.1.3 — Name your secret

1. **Secret name:** type `warehouse-assistant/api-keys`
2. **Description:** type `API keys for the Warehouse Assistant project`
3. Click **"Next"**

#### 2.1.4 — Skip rotation

1. AWS asks if you want automatic rotation. Leave it OFF (default).
2. Click **"Next"**

#### 2.1.5 — Review and save

1. Scroll to the bottom
2. Click the orange **"Store"** button
3. ✅ Your API key is now safely stored in the cloud!

#### 2.1.6 — Copy the secret's ARN (important!)

1. You'll be sent back to the secrets list
2. Click on the secret you just created: `warehouse-assistant/api-keys`
3. At the top you'll see **"Secret ARN"** — it looks like `arn:aws:secretsmanager:us-east-1:123456789012:secret:warehouse-assistant/api-keys-AbCdEf`
4. Click the little **copy icon** next to it
5. **Paste it somewhere safe** (a text file on your computer). You'll need it later.

---

### 🗄️ Step 2.2 — Create an S3 Bucket for Your Data

#### 2.2.1 — Open S3

1. Click the **search bar** at the top of the console
2. Type **"S3"**
3. Click the first result: **"S3"**

#### 2.2.2 — Create a new bucket

1. You'll see a page called "Amazon S3". Click the orange **"Create bucket"** button
2. **Bucket name:** type something unique like `warehouse-assistant-data-[your-initials]-[random-number]`
   - Example: `warehouse-assistant-data-js-47281`
   - ⚠️ Bucket names must be **globally unique** across all of AWS — if taken, try a different number
3. **AWS Region:** make sure it matches the region you picked in Step 1.9
4. Scroll down to **"Block Public Access settings for this bucket"**
5. ✅ Leave **"Block all public access"** TICKED (this is the safe setting)
6. Scroll down more
7. Under **"Bucket Versioning"**, select **"Enable"** (this keeps old versions of files — useful if you mess up)
8. Leave everything else at the default
9. Scroll to the bottom and click orange **"Create bucket"**

✅ You should see a green bar at the top saying "Successfully created bucket".

#### 2.2.3 — Make folders inside the bucket

1. Click on your new bucket name in the list
2. Click the orange **"Create folder"** button
3. **Folder name:** type `data`
4. Click **"Create folder"**
5. Repeat four more times, creating folders named:
   - `processed`
   - `raw`
   - `chroma_backup`
   - `logs`

Your bucket should now look like this:
```
warehouse-assistant-data-js-47281/
├── data/
├── processed/
├── raw/
├── chroma_backup/
└── logs/
```

#### 2.2.4 — Write down your bucket name

In a safe text file, write down:
- **Bucket name:** `warehouse-assistant-data-js-47281` (use YOUR name)
- **Bucket ARN:** `arn:aws:s3:::warehouse-assistant-data-js-47281`

---

### 👮 Step 2.3 — Create an IAM Role (The Security Guard)

An IAM role is a set of permissions. Our EC2 server will "wear" this role like a uniform, and the uniform will say "this server is allowed to read secrets and use S3."

#### 2.3.1 — Open IAM

1. Click the **search bar** at the top
2. Type **"IAM"**
3. Click the first result: **"IAM"**

#### 2.3.2 — Create a policy first (what the role is allowed to do)

1. In the **left sidebar**, click **"Policies"**
2. Click the orange **"Create policy"** button at the top right
3. You'll see two tabs: **"Visual"** and **"JSON"**. Click **"JSON"**
4. Delete everything in the JSON box
5. Paste this in (replace the ARNs with YOUR real bucket and secret ARN):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadSecrets",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:123456789012:secret:warehouse-assistant/api-keys-*"
    },
    {
      "Sid": "S3DataAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::warehouse-assistant-data-js-47281",
        "arn:aws:s3:::warehouse-assistant-data-js-47281/*"
      ]
    },
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

6. Click orange **"Next"** at bottom
7. **Policy name:** type `WarehouseAssistantPolicy`
8. **Description:** type `Allows EC2 to access S3, Secrets Manager, Bedrock, and CloudWatch`
9. Scroll down, click orange **"Create policy"**

✅ Policy created.

#### 2.3.3 — Create the role

1. In the **left sidebar**, click **"Roles"**
2. Click the orange **"Create role"** button
3. Under **"Trusted entity type"**, select **"AWS service"**
4. Under **"Use case"**, select **"EC2"** from the dropdown
5. Click orange **"Next"**
6. You'll see a long list of policies. In the search box, type `WarehouseAssistantPolicy`
7. Tick the checkbox next to your policy
8. Click orange **"Next"**
9. **Role name:** type `WarehouseAssistantRole`
10. **Description:** type `Role for EC2 instances running the Warehouse Assistant`
11. Scroll down, click orange **"Create role"**

✅ Role created. Write its name down: `WarehouseAssistantRole`

---

## 💻 PHASE 3 — Compute (Run the App on EC2)

> ⏱️ Time: 3 hours
> 💰 Cost: FREE for 12 months (t2.micro free tier)

EC2 is like renting a computer that lives in an Amazon data centre. It runs 24/7 so your chatbot is always available.

---

### Step 3.1 — Launch an EC2 instance

#### 3.1.1 — Open EC2

1. Click the **search bar** at the top
2. Type **"EC2"**
3. Click the first result: **"EC2"**

#### 3.1.2 — Start the launch wizard

1. Look for a big orange button that says **"Launch instance"**. Click it.

#### 3.1.3 — Name your instance

1. In the **"Name and tags"** box, type: `warehouse-assistant-server`

#### 3.1.4 — Choose the operating system

1. Under **"Application and OS Images (Amazon Machine Image)"**, you'll see tabs
2. Click the **"Quick Start"** tab
3. Select **"Amazon Linux"** (it should have a label saying **"Free tier eligible"**)
4. Leave the AMI at the default (Amazon Linux 2023)
5. Architecture: leave at **64-bit (x86)**

#### 3.1.5 — Choose instance type

1. Under **"Instance type"**, click the dropdown
2. Select **"t2.micro"** (it will say **"Free tier eligible"** next to it)

#### 3.1.6 — Create a key pair (like a house key for your server)

1. Under **"Key pair (login)"**, click **"Create new key pair"**
2. **Key pair name:** type `warehouse-assistant-key`
3. **Key pair type:** RSA
4. **Private key file format:** choose **.pem** (for Mac/Linux) or **.ppk** (for Windows with PuTTY)
5. Click orange **"Create key pair"**
6. A file will download automatically — **SAVE IT CAREFULLY**. You can't download it again!
7. Move the file to a safe folder like `~/aws-keys/`

#### 3.1.7 — Network settings (firewall)

1. Under **"Network settings"**, click the **"Edit"** button on the right
2. Leave VPC and Subnet at default
3. **Auto-assign public IP:** Enable
4. Under **"Firewall (security groups)"**, select **"Create security group"**
5. **Security group name:** type `warehouse-assistant-sg`
6. **Description:** type `Firewall for warehouse assistant`
7. You'll see inbound rules. Add these:
   - **Rule 1:** Type: SSH, Source type: My IP (this lets only YOU connect)
   - Click **"Add security group rule"**
   - **Rule 2:** Type: Custom TCP, Port range: 5001, Source type: Anywhere (so users can use the chat)
   - Click **"Add security group rule"**
   - **Rule 3:** Type: HTTP, Source type: Anywhere (for a nicer URL later)

#### 3.1.8 — Storage

1. Under **"Configure storage"**, leave it at **8 GiB gp3** (free tier allows up to 30 GiB)

#### 3.1.9 — Advanced details — attach the IAM role

1. Scroll to **"Advanced details"** and click to expand it
2. Scroll down until you find **"IAM instance profile"**
3. From the dropdown, select **`WarehouseAssistantRole`** (the role we made in Step 2.3.3)

#### 3.1.10 — Add startup script (installs Python automatically)

1. Still in "Advanced details", scroll to **"User data"**
2. Paste this into the big text box:

```bash
#!/bin/bash
yum update -y
yum install -y python3.11 python3.11-pip git
cd /home/ec2-user
git clone https://github.com/ozzy2438/Chatbot-Warehouse-Assistant-Langchain-.git warehouse-assistant
chown -R ec2-user:ec2-user warehouse-assistant
cd warehouse-assistant
pip3.11 install -r requirements.txt
pip3.11 install boto3 langchain-aws watchtower
```

3. This script runs automatically when the server starts for the first time.

#### 3.1.11 — Launch!

1. Look at the right sidebar — there's a **"Summary"** panel
2. Click the orange **"Launch instance"** button
3. Wait 30 seconds
4. Click **"View all instances"**
5. Your new server appears! Wait until:
   - **Instance state:** Running (green)
   - **Status check:** 2/2 checks passed

#### 3.1.12 — Get your server's IP address

1. Click on your instance in the list
2. In the details panel at the bottom, find **"Public IPv4 address"**
3. Copy it down — it looks like `54.123.45.67`
4. **Write it down.** This is your server's address on the internet.

---

### Step 3.2 — Connect to Your Server (First Time)

#### 3.2.1 — Open your computer's terminal

- **Mac:** Open "Terminal" (Cmd+Space, type "Terminal")
- **Windows:** Open "PowerShell" or "Windows Terminal"
- **Linux:** Open any terminal

#### 3.2.2 — Secure your key file

```bash
chmod 400 ~/aws-keys/warehouse-assistant-key.pem
```

#### 3.2.3 — Connect to the server

Replace `54.123.45.67` with YOUR server's IP:

```bash
ssh -i ~/aws-keys/warehouse-assistant-key.pem ec2-user@54.123.45.67
```

1. It will ask *"Are you sure you want to continue connecting?"* — type `yes` and press Enter
2. You're now inside your AWS server! You'll see a prompt like `[ec2-user@ip-172-31-... ~]$`

#### 3.2.4 — Check that the startup script finished

```bash
cd warehouse-assistant
ls
```

You should see all your project files. If not, wait 2 more minutes and try again — the startup script may still be running.

---

### Step 3.3 — Set up the .env on the Server

Instead of storing the `.env` file on the server (unsafe!), we'll pull the secret from Secrets Manager at runtime.

#### 3.3.1 — Create a helper script

Still connected to the server via SSH, run:

```bash
nano load_secrets.py
```

Paste this in:

```python
"""Fetches API keys from AWS Secrets Manager and sets them as environment variables."""
import os
import json
import boto3

SECRET_NAME = "warehouse-assistant/api-keys"
REGION = "us-east-1"  # Change to your region if different

def load_secrets():
    client = boto3.client("secretsmanager", region_name=REGION)
    response = client.get_secret_value(SecretId=SECRET_NAME)
    secret_dict = json.loads(response["SecretString"])
    for key, value in secret_dict.items():
        os.environ[key] = value
    print(f"Loaded {len(secret_dict)} secrets from AWS Secrets Manager")

if __name__ == "__main__":
    load_secrets()
```

Press **Ctrl+O**, then Enter (to save), then **Ctrl+X** (to exit nano).

#### 3.3.2 — Test it

```bash
python3.11 load_secrets.py
```

You should see: `Loaded 2 secrets from AWS Secrets Manager`. If you see an error, check:
- Is your region correct in the script?
- Did you attach the IAM role to the instance?

---

### Step 3.4 — Run the App as a Service

Services are programs that restart automatically if they crash. Let's make our chatbot one.

#### 3.4.1 — Create a service file

```bash
sudo nano /etc/systemd/system/warehouse-assistant.service
```

Paste this in:

```ini
[Unit]
Description=Warehouse Assistant Flask App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/warehouse-assistant
ExecStart=/usr/bin/python3.11 app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Save with **Ctrl+O**, Enter, **Ctrl+X**.

#### 3.4.2 — Start the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable warehouse-assistant
sudo systemctl start warehouse-assistant
sudo systemctl status warehouse-assistant
```

You should see green text saying "active (running)".

#### 3.4.3 — Test it from your laptop

Open a web browser on your laptop. Go to:

```
http://YOUR-SERVER-IP:5001
```

(Replace `YOUR-SERVER-IP` with the IP from Step 3.1.12.)

🎉 **Your chatbot is live on the internet!**

---

### Step 3.5 — Set Up CloudWatch Logs

So you can see logs in AWS instead of having to SSH in every time.

#### 3.5.1 — Install CloudWatch agent

Still in SSH:

```bash
sudo yum install -y amazon-cloudwatch-agent
```

#### 3.5.2 — Configure it

```bash
sudo nano /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
```

Paste:

```json
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/home/ec2-user/warehouse-assistant/etl_orchestrator.log",
            "log_group_name": "warehouse-assistant",
            "log_stream_name": "etl-{instance_id}"
          }
        ]
      }
    }
  }
}
```

Save and exit.

#### 3.5.3 — Start the agent

```bash
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
```

#### 3.5.4 — View logs in AWS Console

1. Back in the AWS Console, search for **"CloudWatch"** and click it
2. In the left sidebar, click **"Log groups"**
3. You'll see `warehouse-assistant` — click it
4. Click any log stream to see real-time logs

---

## ⚙️ PHASE 4 — Automation (EventBridge + Lambda)

> ⏱️ Time: 2 hours
> 💰 Cost: FREE

Goal: automatically run the ETL pipeline every Monday at 2 AM without pressing any button.

---

### Step 4.1 — Create a Lambda Function

#### 4.1.1 — Open Lambda

1. Search for **"Lambda"** in the AWS console and click it
2. Click orange **"Create function"**

#### 4.1.2 — Basic info

1. Select **"Author from scratch"**
2. **Function name:** `trigger-warehouse-etl`
3. **Runtime:** Python 3.11
4. **Architecture:** x86_64
5. Under **"Permissions"**, expand "Change default execution role"
6. Select **"Create a new role with basic Lambda permissions"**
7. Click orange **"Create function"**

#### 4.1.3 — Replace the code

1. You're now looking at the code editor. Delete everything in `lambda_function.py`
2. Paste this:

```python
import boto3
import os

EC2_INSTANCE_ID = os.environ["EC2_INSTANCE_ID"]

def lambda_handler(event, context):
    ssm = boto3.client("ssm")
    response = ssm.send_command(
        InstanceIds=[EC2_INSTANCE_ID],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": [
                "cd /home/ec2-user/warehouse-assistant",
                "sudo -u ec2-user python3.11 etl_orchestrator.py >> etl_orchestrator.log 2>&1"
            ]
        },
        TimeoutSeconds=7200
    )
    return {
        "statusCode": 200,
        "commandId": response["Command"]["CommandId"]
    }
```

3. Click **"Deploy"** (the orange button above the editor)

#### 4.1.4 — Add the instance ID environment variable

1. Click the **"Configuration"** tab (near the top)
2. In the left menu, click **"Environment variables"**
3. Click **"Edit"**
4. Click **"Add environment variable"**
5. **Key:** `EC2_INSTANCE_ID`
6. **Value:** your EC2 instance ID (find it on the EC2 page — looks like `i-0a1b2c3d4e5f6g7h8`)
7. Click **"Save"**

#### 4.1.5 — Give Lambda permission to control EC2 via SSM

1. Still in Configuration tab
2. In left menu, click **"Permissions"**
3. Click the **Role name link** (opens IAM in a new tab)
4. Click **"Add permissions"** → **"Attach policies"**
5. In the search box, type `AmazonSSMFullAccess`
6. Tick it, then click **"Add permissions"**

---

### Step 4.2 — Schedule with EventBridge

#### 4.2.1 — Open EventBridge

1. Search for **"EventBridge"** and click it
2. In the left sidebar, click **"Rules"**
3. Click orange **"Create rule"**

#### 4.2.2 — Rule details

1. **Name:** `weekly-etl-trigger`
2. **Description:** `Runs the ETL pipeline every Monday at 2 AM`
3. Under **"Rule type"**, select **"Schedule"**
4. Click **"Continue to create rule"**

#### 4.2.3 — Schedule

1. Select **"A schedule that runs at a regular rate, such as every 10 minutes"** — wait, no. Select **"A fine-grained schedule that runs at a specific time"**
2. In the cron expression box, type: `cron(0 2 ? * MON *)`
   - This means: minute 0, hour 2, any day of month, any month, Monday
3. Click **"Next"**

#### 4.2.4 — Target

1. **Target types:** AWS service
2. **Select a target:** Lambda function
3. **Function:** `trigger-warehouse-etl`
4. Click **"Next"**
5. Skip tags, click **"Next"**
6. Review and click **"Create rule"**

✅ Every Monday at 2 AM, your ETL pipeline runs automatically.

---

## 🧠 PHASE 5 — Amazon Bedrock (The Second AI Brain)

> ⏱️ Time: 3 hours
> 💰 Cost: Pennies (Claude 3 Haiku ~$0.25/1M tokens)

Now the star of the show — we add Amazon Bedrock as an alternative AI backend.

---

### Step 5.1 — Request Access to Bedrock Models

#### 5.1.1 — Open Bedrock

1. Search for **"Bedrock"** and click it
2. You may be asked to accept terms — click **"Agree"** / **"Get started"**

#### 5.1.2 — Request model access

1. In the **left sidebar**, scroll down to find **"Model access"** and click it
2. Click orange **"Modify model access"** (top right)
3. You'll see a long list of AI models
4. Tick the checkboxes for:
   - **Anthropic → Claude 3 Haiku**
   - **Amazon → Titan Text Embeddings V2**
5. Click **"Next"**
6. For Claude, you'll be asked about use case — tick "Internal employee productivity" and fill in a short description like *"Testing Bedrock for a portfolio project"*
7. Click **"Submit"**
8. Wait 1-5 minutes. The status will change from "In Progress" to "Access granted" (refresh the page to check).

---

### Step 5.2 — Test Bedrock Works

#### 5.2.1 — Use the Bedrock playground

1. In Bedrock, left sidebar, click **"Chat / Text"** (under "Playgrounds")
2. Click **"Select model"** → Anthropic → Claude 3 Haiku
3. Type: `Hello, say hi back in one word.`
4. Click **"Run"**
5. You should see a response. ✅ Bedrock is working.

---

### Step 5.3 — Modify Your Code to Support Bedrock

SSH into your EC2 server:

```bash
ssh -i ~/aws-keys/warehouse-assistant-key.pem ec2-user@YOUR-SERVER-IP
cd warehouse-assistant
```

#### 5.3.1 — Install the Bedrock LangChain library

```bash
pip3.11 install langchain-aws
```

#### 5.3.2 — Edit chatbot.py

```bash
nano chatbot.py
```

Find the lines that look like this (around line 107):

```python
self.llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)
self.embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

Replace them with:

```python
provider = os.getenv("LLM_PROVIDER", "openai")

if provider == "bedrock":
    from langchain_aws import ChatBedrock, BedrockEmbeddings
    self.llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"temperature": 0.1, "max_tokens": 2000},
        region_name="us-east-1",  # match your region
    )
    self.embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
    )
    print("[AI] Using Amazon Bedrock (Claude 3 Haiku + Titan Embeddings)")
else:
    self.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
    )
    self.embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("[AI] Using OpenAI (GPT-4o-mini + text-embedding-3-small)")
```

Save with Ctrl+O, Ctrl+X.

#### 5.3.3 — Switching between providers

The LLM provider is controlled by the `LLM_PROVIDER` environment variable, which is stored in Secrets Manager. To switch:

1. Go back to the AWS Console, search **"Secrets Manager"**
2. Click `warehouse-assistant/api-keys`
3. Click **"Retrieve secret value"** → **"Edit"**
4. Change `LLM_PROVIDER` value from `openai` to `bedrock` (or back)
5. Click **"Save"**
6. SSH to your server and restart the app: `sudo systemctl restart warehouse-assistant`

✨ **You can now switch AI brains by changing one word in the AWS Console.**

---

## 🛠️ Code Changes You Will Make

Here is the complete summary of all the code modifications across the whole migration:

### File 1: `requirements.txt`
**Add** these lines at the bottom:
```
boto3>=1.28.0
langchain-aws>=0.1.0
watchtower>=3.0.0
```

### File 2: `chatbot.py`
Around line 21, add at the top:
```python
# Load secrets from AWS Secrets Manager if running on AWS
try:
    from load_secrets import load_secrets
    load_secrets()
except Exception:
    # Fallback to local .env
    from dotenv import load_dotenv
    load_dotenv()
```

Replace the LLM initialisation block (see Step 5.3.2 above).

### File 3: `etl_orchestrator.py`
Add a function to upload outputs to S3 after writing them locally:

```python
import boto3

S3_BUCKET = os.getenv("S3_BUCKET", "warehouse-assistant-data-js-47281")

def upload_to_s3(local_path: Path, s3_key: str):
    """Copy a local file up to S3."""
    try:
        s3 = boto3.client("s3")
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)
        print(f"  ↑ S3: {s3_key}")
    except Exception as e:
        print(f"  ⚠ S3 upload failed: {e}")
```

Then call it after each `to_csv`:
```python
df_final.to_csv(files['final'], index=False)
upload_to_s3(files['final'], f"data/final_product_database.csv")
```

### File 4: `load_secrets.py` (NEW FILE)
Created in Step 3.3.1.

### File 5: `app.py`
Replace the hardcoded secret key:
```python
# BEFORE
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

# AFTER
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())
```

---

## ✅ Testing Everything Works

Run these checks in order:

### Test 1 — The app is reachable
Open `http://YOUR-SERVER-IP:5001` in a browser. You should see the chat UI.

### Test 2 — The chatbot responds
Type: *"How many products are in Sydney?"*
You should get a real answer in under 3 seconds.

### Test 3 — S3 has data
Go to S3 console → your bucket → `data/` folder. You should see CSV files.

### Test 4 — Bedrock mode works
1. Switch `LLM_PROVIDER` to `bedrock` in Secrets Manager
2. SSH in: `sudo systemctl restart warehouse-assistant`
3. Ask the chatbot a question — it should still answer (now using Claude 3 Haiku)

### Test 5 — Scheduled ETL triggers
1. Go to Lambda → `trigger-warehouse-etl`
2. Click **"Test"** → create a dummy test event → click **"Test"** again
3. Check CloudWatch Logs → `warehouse-assistant` log group — you should see ETL activity

### Test 6 — CloudWatch sees logs
Navigate to CloudWatch → Log groups → `warehouse-assistant`. Logs should be flowing in.

---

## 🚨 Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| Can't SSH into EC2 | Security group blocks your IP | Check security group Inbound rule "SSH" allows your current IP |
| "Access denied" when fetching secret | IAM role not attached | EC2 → Instance → Actions → Security → Modify IAM role |
| Chatbot returns blank answers | No data in ChromaDB | SSH in, run `python3.11 etl_orchestrator.py --test` |
| Bedrock call fails with "AccessDenied" | Model access not granted | Bedrock → Model access → Request access to Claude 3 Haiku |
| Flask app crashes on start | Missing dependency | SSH in, run `pip3.11 install -r requirements.txt` |
| S3 uploads silently fail | Wrong bucket name in env | Check `S3_BUCKET` env variable matches your real bucket |
| Lambda times out | Script runs longer than 15 min | Lambda only triggers — it doesn't run the ETL itself. Make sure it's using SSM SendCommand |

---

## 💰 Cost Calculator

**Free Tier (first 12 months):**
| Service | Free Allowance | Your Usage | Cost |
|---|---|---|---|
| EC2 t2.micro | 750 hours/month | 744 hours (always on) | $0 |
| S3 | 5 GB storage + 20,000 GETs | ~200 MB, few thousand GETs | $0 |
| Lambda | 1M requests + 400k GB-sec | 4 triggers/month | $0 |
| EventBridge | Unlimited scheduled rules | 1 rule | $0 |
| CloudWatch Logs | 5 GB ingestion | ~100 MB | $0 |
| Data Transfer | 100 GB out/month | likely <1 GB | $0 |

**Always paid:**
| Service | Cost |
|---|---|
| Secrets Manager | $0.40/secret/month |
| Bedrock (Claude 3 Haiku) | $0.25 per 1M input tokens, $1.25 per 1M output (demo = pennies) |

**Expected monthly bill during portfolio demo:** **Less than $1.**
**Expected monthly bill after free tier (month 13+):** **Around $10.**

---

## 🏆 What You Have Achieved

By the end of this guide, you can honestly say on a CV or in an interview:

✅ Designed and deployed a production AI system on AWS
✅ Implemented secure secret management with AWS Secrets Manager
✅ Used IAM roles and least-privilege policies (no credentials on disk)
✅ Built a serverless automation pipeline with EventBridge + Lambda + SSM
✅ Deployed a Flask/Socket.IO server on EC2 with systemd
✅ Integrated Amazon Bedrock (Claude 3 Haiku + Titan Embeddings) into a LangChain RAG pipeline
✅ Built a provider-agnostic LLM architecture switchable via environment variable
✅ Set up centralised observability with CloudWatch Logs
✅ Used S3 as a data lake for ETL outputs
✅ Kept the total monthly cost under $1 through free tier optimisation

---

## 🗺️ Where to Go Next

After this guide, natural next steps that would further boost your CV:

1. **Add a custom domain with Route 53** — `chatbot.yourname.com` instead of an IP
2. **Add HTTPS with AWS Certificate Manager** — professional sites use SSL
3. **Migrate ChromaDB to Amazon OpenSearch Serverless** — full AWS-native RAG
4. **Add CloudWatch Alarms** — email you if the chatbot goes down
5. **Put everything in Terraform** — Infrastructure as Code is a huge CV bonus
6. **Add GitHub Actions CI/CD** — auto-deploy on `git push`

Take it one step at a time. Congratulations on finishing the guide! 🎉
