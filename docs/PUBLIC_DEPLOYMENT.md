# Deployment Guide for RAG Anything

To make this application work independently of your laptop, you need to deploy it to a cloud provider. We recommend **Render.com** or **Railway.app** as they are easy to use and have free tiers.

## Prerequisites
1. A **GitHub Account**.
2. A **Supabase Account** (you already have this).
3. A **HuggingFace Account** (you already have this).

## Step 1: Push Code to GitHub
1. Create a new repository on GitHub.
2. Push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

## Step 2: Deploy on Render.com
1. Sign up at [Render.com](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository.
4. **Settings:**
   - **Name:** rag-chatbot
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python webapp_api/full_rag_server.py`
5. **Environment Variables:**
   Add the following variables (copy values from your local `.env` file):
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `SUPABASE_DB_PASSWORD`
   - `HUGGINGFACE_API_KEY`
   - `LLM_PROVIDER` = `huggingface`
   - `EMBEDDING_PROVIDER` = `huggingface`

6. Click **Create Web Service**.

## Step 3: Access Anywhere
Render will give you a public URL (e.g., `https://rag-chatbot.onrender.com`). You can access this URL from any device, anywhere in the world.

## Important Note on Data
Since we changed the embedding provider to HuggingFace for the cloud version, **existing documents uploaded from your laptop might not be searchable** due to incompatible vector dimensions.
- You might need to re-upload documents in the new cloud deployment.
