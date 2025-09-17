# Document-Question-Answering
Document question-answering using NVIDIA NIMs

pdf-qa/
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ config.yaml
├─ documents/          # <— customer drops PDFs here
│  └─ .gitkeep
└─ vectorstore/        # <— persisted FAISS index
   └─ .gitkeep
