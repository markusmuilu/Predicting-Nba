## 🏀 Predict NBA — Machine Learning NBA Game Predictor

FastAPI • Supabase • Scikit-Learn • Docker • ESPN Automation

This project predicts NBA game outcomes using machine learning.
It automatically:

- Collects team game logs  
- Cleans + feature-engineers datasets  
- Trains a prediction model  
- Generates daily predictions  
- Updates results using ESPN  
- Stores prediction history in Supabase  
- Exposes a clean FastAPI API  


You can run it using Docker (recommended) or by installing it locally with pip.

## 📦 Features

- 🧠 ML-based game winner predictions  
- ⚙️ Automated data collection & cleaning  
- 🗄 Supabase integration (tables + bucket)  
- 📅 Daily predictions using ESPN’s scoreboard  
- 🌐 FastAPI backend with Swagger docs  
- 🐳 Dockerized deployment (best option)  
- 🐍 Optional local installation with pyproject.toml  
- ⭐ Production-ready structure for real-world use  

# 📁 Project Structure

`````
src/
│── nba_predict/
│   ├── backend/
│   │   ├── main.py
│   │   └── routes/
│   │       ├── predict.py
│   │       └── update.py
│   │
│   ├── features/
│   │   ├── data_collector.py
│   │   ├── data_cleaner.py
│   │   ├── model_trainer.py
│   │   ├── model_predictor.py
│   │   └── daily_predictor.py
│   │
│   └── utils/
│       ├── logger.py
│       └── exception.py
│
Dockerfile
docker-compose.yml
pyproject.toml
requirements.txt
.env.example
setup_project.py
`````
# 🔧 Installation Options

You can run the project in two ways:

# 🐳 Option 1 — Run With Docker (Recommended)
**1️⃣ Create your .env**

Copy the example:
`````
cp .env.example .env
`````

Fill in:
`````
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_service_or_anon_key
`````
# 2️⃣ Start the API
`````
docker compose up --build -d
`````

API is now running at:

➡️ http://localhost:8000

➡️ http://localhost:8000/docs
 (Swagger UI)

# 3️⃣ First-time setup (create tables + train model)
`````
docker run --env-file .env predict-nba_api python setup_project.py
`````

# 🐍 Option 2 — Local Python Installation
1️⃣ Create virtual environment
`````
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
`````

# 2️⃣ Install project
`````
pip install .
`````

or:
`````
pip install -r requirements.txt
`````
# 3️⃣ Initialize Supabase + train model
`````
python setup_project.py
`````
# 4️⃣ Start API
`````
uvicorn src.backend.main:app --reload
`````
# 📡 API Endpoints
Predict matchup
`````
GET /predict?team1=CLE&team2=ATL
`````

Example response:
`````
{
  "winner": "CLE",
  "confidence": 73.5
}
`````
# Daily update (ESPN results + new predictions)
`````
POST /update
`````

Runs:

- Update finished game
- Insert new predictions for today

# 🗄 Supabase schema
Tables
teams
id (int) | name (text)

current_predictions

Stores today's predictions.

prediction_history
    
Stores historical predictions + correctness.

**Bucket**: modelData

Contains:

raw logs

cleaned CSVs

model file (prediction_model.skops)

# 🌩 Deploy to AWS EC2 (with Docker)

Install Docker:
`````
sudo apt update
sudo apt install docker.io docker-compose -y
`````

Clone repo:
`````
git clone https://github.com/your/repo.git
cd repo
`````

Run:
`````
docker compose up --build -d
`````

Done — your API is live.



# ⭐ If you like this project, consider giving it a GitHub star!

