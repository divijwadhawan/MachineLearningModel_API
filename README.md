# MachineLearningModel_API
Simple Machine Learning Model making available as an API

## Create Virtual Environment
python3 -m venv venv

## Activate Venv
source venv/bin/activate

## Install Requirements
pip3 install -r requirements.txt

## Train the Model and save it as a .pkl file
python3 train_model.py

## Run the API
python3 api.py

## Use python3 uvicorn to run the server
python3 -m uvicorn main:app --reload

## Server is Live
http://127.0.0.1:8000/docs

## Use Postman to send a request
URL : http://127.0.0.1:8000/predict

JSON :
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}


# To deploy on Render

## Create start command for Render
Make a file start.sh:

#!/bin/bash
uvicorn api:app --host 0.0.0.0 --port $PORT


Then make it executable:
chmod +x start.sh


## Push code to GitHub.
Your repo should have:

api.py
train_model.py
decision_tree_model.pkl
requirements.txt
start.sh


## Connect repo to Render → New Web Service.

Runtime: Python

Build Command:

pip install -r requirements.txt && python train_model.py


Start Command:

./start.sh


## Render will give you a public URL like:
👉 https://your-app.onrender.com/predict
