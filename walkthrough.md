# Offline KYC System - Setup Guide (Split Stack)

We have separated the system to run the lightweight Node.js backend on **Render** (Free Tier) and the heavy Python AI Engine on **Hugging Face Spaces** (Free CPU Tier).

## 1. Setup Hugging Face Space (Python Engine)
1.  Go to Hugging Face Spaces and create a new Space.
    *   **SDK**: Select **Docker**.
    *   **Hardware**: Select **Free**.
    *   **Visibility**: **Public** (Easiest).
2.  Upload the files from the `hf_kyc_space/` folder to your Space:
    *   `app.py`
    *   `Dockerfile`
    *   `requirements.txt`
3.  Wait for the Space to build. Once running, copy the **Direct URL** (e.g., `https://username-spacename.hf.space`).

## 2. Setup Render (Node.js Backend)
1.  Go to your Render Dashboard -> Your Service.
2.  **Environment Variables**:
    *   Add `KYC_API_URL` -> Paste your Hugging Face Space URL (e.g., `https://username-spacename.hf.space`).
3.  **Build Command**:
    *   Set simply to: `npm install` (We NO LONGER need pip install here).
4.  **Start Command**:
    *   `node index.js`
5.  Deploy!

## 3. API Usage (Same as before)
Your endpoints remain exactly the same. the `index.js` automatically routes the heavy work to your Hugging Face space.

### A. CNIC Verification
**POST** `/api/vision/verify-cnic`
```json
{ "cnicFrontBase64": "..." }
```

### B. Face Verification
**POST** `/api/kyc/face`
```json
{ "cnicImage": "...", "selfieImage": "..." }
```

### C. Shop Verification
**POST** `/api/kyc/shop`
```json
{ "shopImage": "..." }
```
