# 👋 Hi, I'm Alfred So (蘇哲緯)

### ML Engineer | 9 Cloud & ML Certifications | Published Author | Production MLOps

I build **production-ready machine learning systems** deployed at scale. Specializing in RAG systems, multi-agent AI, vision-language models, neural recommendation systems, and real-time ML pipelines—**all with live demos you can try right now**.

🎯 **Seeking ML Engineer / MLOps / AI Engineer roles in Hong Kong & San Francisco Bay Area** | Available Jun 2026

---

## 🚀 What I Build

- 🤖 **Production ML Systems**: Neural recommenders, RAG chatbots, multi-agent AI, fraud detection, time series forecasting
- 🧠 **AI/LLM Engineering**: LangChain, CrewAI, vision-language models (PaliGemma), fine-tuning with LoRA/PEFT
- ☁️ **Cloud MLOps**: AWS (SageMaker, Lambda), GCP, Azure, Databricks, Modal Labs, serverless deployment
- 📊 **Deep Learning**: PyTorch, TensorFlow, Neural Collaborative Filtering, LSTM, Transformers
- 🛠️ **Tech Stack**: Python, FastAPI, Streamlit, Docker, REST APIs, CI/CD pipelines

**Unique Edge:** 
- 📚 Published author of ["The Technocratic Dividend"](https://www.amazon.com/dp/9887144614) (2024, Amazon, Barnes & Noble)—demonstrates ability to communicate complex technical concepts
- 🏥 4+ years healthcare domain expertise as licensed Occupational Therapist—translating complex stakeholder needs into technical solutions

---

## 🏆 Certifications (9 Total)

**Advanced ML & AI:**
[![AWS ML](https://img.shields.io/badge/AWS-ML_Specialty-FF9900?style=flat-square&logo=amazon-aws)](https://aws.amazon.com/certification/)
[![Azure AI](https://img.shields.io/badge/Azure-AI_Engineer-0078D4?style=flat-square&logo=microsoft-azure)](https://learn.microsoft.com/en-us/certifications/)
[![GCP ML](https://img.shields.io/badge/GCP-ML_Engineer-4285F4?style=flat-square&logo=google-cloud)](https://cloud.google.com/certification)
[![Databricks](https://img.shields.io/badge/Databricks-ML_Professional-FF3621?style=flat-square&logo=databricks)](https://www.databricks.com/learn/certification)
[![NVIDIA AI Infra](https://img.shields.io/badge/NVIDIA-AI_Infrastructure-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/en-us/training/)
[![NVIDIA GenAI](https://img.shields.io/badge/NVIDIA-Generative_AI-76B900?style=flat-square&logo=nvidia)](https://www.nvidia.com/en-us/training/)

**Foundations:**  
✅ DeepLearning.AI TensorFlow Developer  
✅ IBM Data Science Professional  
✅ Google Data Analytics Professional

---

## 💻 Featured Projects (All Live!)

### 🎬 [MovieMatch - Neural Collaborative Filtering Recommender](https://github.com/Donald8585/moviematch-recsys)
Deep learning recommendation system with **66.2% improvement** over baseline (RMSE: 0.93 vs 2.76)
- 🧠 **Neural CF architecture** in PyTorch: 50-dim embeddings, [64→32→16] MLP with BatchNorm & dropout
- 📊 Trained on **MovieLens 100K** (100K ratings, 943 users, 1,682 movies) with 93.7% sparsity
- ⚡ **FastAPI REST API** with <1s inference, automatic OpenAPI docs, Pydantic validation
- 🎯 Production-ready with health checks, model persistence, real-time personalized recommendations
- **[Live Demo →](https://huggingface.co/spaces/Donald8585/moviematch-recsys)** | Try movie recommendations!

### 🤖 [Multi-Agent AI Research Assistant](https://github.com/Donald8585/ai-research-assistant)
Production multi-agent system orchestrating **3 specialized AI agents** (Researcher, Writer, Critic)
- 🔄 **CrewAI framework** with sequential workflow, quality assurance, and fact-checking
- 🌐 **Real-time web search** via SerperDev API (2,500 free searches/month) for current 2026 info
- 🧠 **Cohere command-r7b-12-2024** via native LLM integration—zero costs using free tier (1,000 calls/month)
- 📝 Generates comprehensive markdown reports with live citations in 60-90 seconds
- ⚡ Deployed on **Hugging Face Spaces** (CPU-basic tier) with Streamlit UI
- **[Live Demo →](https://huggingface.co/spaces/Donald8585/ai-research-assistant)** | Generate AI research reports!

### 🏥 [HK Healthcare RAG Chatbot](https://github.com/Donald8585/hk-healthcare-rag-chatbot)
Production RAG system for Hong Kong healthcare queries with **3-5s latency** and source attribution
- 📚 **616 HK healthcare documents** (1,785 chunks) in ChromaDB vector database
- 🤖 **Modern LangChain LCEL** with Cohere command-r7b & embed-english-light-v3
- ⚡ **3-5s query latency** with conversation history, source citations, persistent chat
- 💰 **$0/month operational cost** on Streamlit Cloud + Hugging Face Spaces with free-tier Cohere APIs
- 🔧 Solved production issues: Cohere model deprecation migration, SQLite compatibility via pysqlite3
- **[Live Demo →](https://huggingface.co/spaces/Donald8585/hk-health-rag)** | Ask healthcare questions!

### 🎨 [PaliGemma Vision-Language Model Fine-tuning](https://github.com/Donald8585/paligemma-image-captioning)
Fine-tuned **Google's 3B parameter PaliGemma** model with dual-platform deployment
- 🧠 **LoRA fine-tuning** (rank=8, <1% trainable params) on vision-language model
- ⚡ **<2s inference** after cold start via Modal Labs T4 GPU with persistent volume caching (15GB weights)
- 🚀 **Dual deployment**: HuggingFace ZeroGPU + Modal Labs serverless with CORS-enabled FastAPI endpoints
- 🎯 Generates contextual captions (e.g., "automobile model parked on a street") with 20-token outputs
- 🌐 **Embeddable frontend** on GitHub Pages with drag-and-drop upload, base64 encoding, async fetch API
- **[Live Demo →](https://donald8585.github.io/paligemma-image-captioning/)** | Upload images for AI captions!
- **[HuggingFace Model →](https://huggingface.co/Donald8585/paligemma-caption-finetuned)**

### 📈 [Stock Forecast ML Dashboard](https://github.com/Donald8585/stock-forecast-ml-dashboard)
Real-time **dual-model forecasting system** (LSTM + Prophet) with **<10% MAPE**
- 🧠 **3-layer LSTM** neural network with trend dampening (30% drift reduction vs pure DL)
- 📊 **Twelve Data API** integration (800 free calls/day) with yfinance fallback for real-time data
- ⚡ **20s training time**, sub-second inference with model caching across 7-90 day forecasts
- 📉 **Interactive Plotly visualizations**: candlestick charts, confidence intervals, side-by-side model comparison
- 🎯 Comprehensive evaluation: MAPE, RMSE, MAE on 80/20 splits with 365-730 days historical data
- **[Live Demo →](https://huggingface.co/spaces/Donald8585/time-series-forecast-dashboard)** | Forecast any stock!

### 🔐 [Credit Card Fraud Detection - AWS MLOps](https://github.com/Donald8585/fraud-detection-mlops)
End-to-end ML pipeline deployed on **AWS with serverless architecture**
- 🚀 **AWS SageMaker + Lambda + API Gateway** for real-time inference (<100ms latency)
- ⚡ **99.99% availability** SLA with automated training, hyperparameter tuning, A/B testing
- 📊 **XGBoost model** on 284K transactions achieving 99.9% accuracy and 0.98 AUC-ROC
- 🔄 **CI/CD pipeline** with automated model deployment and monitoring via CloudWatch
- 💰 Scalable serverless infrastructure handling real-time predictions
- **[Live Demo →](https://huggingface.co/spaces/Donald8585/fraud-detection-demo)** | Test fraud detection!
- **[GitHub →](https://github.com/Donald8585/fraud-detection-mlops)**

### 🩺 [Diabetes Prediction Web App](https://github.com/Donald8585/diabetes-prediction-demo)
Interactive ML web app for **real-time diabetes risk prediction**
- 🤖 **XGBoost classifier** on 700K patient dataset (26 features) achieving 68% accuracy, 72% ROC-AUC
- 📊 Feature pipeline handling one-hot encoding (6 categorical variables) + 3 binary medical history flags
- 🎯 Family history identified as dominant predictor (82% feature importance)
- 🌐 Deployed on HuggingFace Spaces with Gradio UI and sample patient profiles
- **[Live Demo →](https://huggingface.co/spaces/Donald8585/diabetes-prediction)** | Check diabetes risk!

---

## 🛠️ Tech Stack

**Machine Learning & AI:**  
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Hugging Face](https://img.shields.io/badge/-Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/-LangChain-000000?style=flat-square&logo=chainlink&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-189FF4?style=flat-square&logo=xgboost&logoColor=white)

**MLOps & Cloud:**  
![AWS](https://img.shields.io/badge/-AWS-232F3E?style=flat-square&logo=amazon-aws)
![Azure](https://img.shields.io/badge/-Azure-0078D4?style=flat-square&logo=microsoft-azure)
![GCP](https://img.shields.io/badge/-GCP-4285F4?style=flat-square&logo=google-cloud)
![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Databricks](https://img.shields.io/badge/-Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white)
![Modal](https://img.shields.io/badge/-Modal_Labs-000000?style=flat-square&logo=modal&logoColor=white)

**Development:**  
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white)

**Data & Databases:**  
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SQL](https://img.shields.io/badge/-SQL-4479A1?style=flat-square&logo=postgresql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/-ChromaDB-FF6B6B?style=flat-square&logo=database&logoColor=white)

---

## 📚 Published Work

### 📖 [The Technocratic Dividend](https://www.amazon.com/dp/9887144614)
**Author** | Published 2024 via Axiom Rising Media | ISBN: 9887144614

244-page analysis of evidence-based governance and technology policy, distributed through Amazon, Barnes & Noble, IngramSpark, and independent bookstores. Demonstrates ability to synthesize complex technical concepts into accessible narratives for general audiences—critical skill for cross-functional ML engineering roles.

📚 [Amazon](https://www.amazon.com/dp/9887144614) | 📚 [Barnes & Noble](https://www.barnesandnoble.com/w/the-technocratic-dividend-alfred-so/1147629550) | 🌐 [Author Website](https://www.alfredso.com)

---

## 📊 GitHub Stats

![Alfred's GitHub Stats](https://github-readme-stats.vercel.app/api?username=Donald8585&show_icons=true&theme=tokyonight&hide_border=true&count_private=true)

![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=Donald8585&layout=compact&theme=tokyonight&hide_border=true)

---

## 🎯 Currently (January 2026)

- 🎓 **MSc Data Science & AI** at Hang Seng University of Hong Kong (graduating May 2026, GPA: 3.8+)
- 🏗️ Building production ML systems: neural recommenders, RAG chatbots, multi-agent AI, vision-language models
- 📚 Certified: AWS ML Specialty, Azure AI Engineer, GCP ML Engineer, Databricks ML Professional, NVIDIA AI (2 certs)
- 💼 Seeking **ML Engineer / MLOps Engineer / AI Engineer** roles in Hong Kong & San Francisco Bay Area
- 🌱 Exploring advanced LLM engineering, model fine-tuning (LoRA/PEFT), serverless ML infrastructure (Modal Labs)

---

## 📫 Let's Connect

[![Portfolio](https://img.shields.io/badge/-Portfolio-000000?style=flat-square&logo=vercel&logoColor=white)](https://www.alfredso.com/portfolio)
[![LinkedIn](https://img.shields.io/badge/-Alfred_So-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/alfred-so)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:fiverrkroft@gmail.com)
[![GitHub](https://img.shields.io/badge/-Donald8585-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Donald8585)
[![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/sword4949)
[![YouTube](https://img.shields.io/badge/-Wait_What_Happened-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/@waitwhathappened-real)

---

## 🌟 Why Work With Me?

✅ **Production-First Mindset:** 7 live demos deployed—not just notebooks, real systems users can try  
✅ **Multi-Cloud Certified:** AWS, Azure, GCP, Databricks, NVIDIA—proven expertise across platforms  
✅ **End-to-End ML:** From data processing → model training → production deployment with MLOps  
✅ **Cost-Conscious Engineering:** Built systems achieving $0/month operational costs via serverless + free tiers  
✅ **Healthcare Domain Expert:** 4+ years OT experience translating complex stakeholder needs into technical solutions  
✅ **Published Technical Writer:** Author of "The Technocratic Dividend"—proven ability to communicate complex concepts  

---

💡 **Open to:** ML Engineer | MLOps Engineer | AI/LLM Engineer | Data Scientist  
📍 **Location:** Hong Kong (current) → San Francisco Bay Area (2027-2028 goal)  
📅 **Available:** June 2026

⭐️ From [Donald8585](https://github.com/Donald8585) | Check out my [7 live ML demos](https://www.alfredso.com/portfolio)!
