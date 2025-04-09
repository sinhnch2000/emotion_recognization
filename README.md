## Step 1: Install Anaconda

Download and install Anaconda from the official website: <https://docs.anaconda.com/anaconda/install/>

## Step 2: Create a new environment

```bash
conda create --name emotion python=3.12.4
conda activate emotion
```

## Step 3: Install requirements

```bash
pip install -r requirements.txt
```

## Step 4: Start the server

```bash
make run-dev
```
Use: http://127.0.0.1:8000 if Linux

or
```bash
fastapi dev main.py --port 8000
```
Use: http://localhost:8000/ if Windows

## Step 5: Test by UI gradio
```bash
python app_test.py
```
Use: http://127.0.0.1:7860/

