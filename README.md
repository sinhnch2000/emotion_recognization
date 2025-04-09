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

## Step 4: Set environment variables

```bash
cp .env.example .env
```

Update `.env` file with the correct values

## Step 5: Start the server
```bash
make run-dev
```
