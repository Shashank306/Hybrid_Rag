# main.py
"""
Entry point for running the FastAPI app for Hybrid RAG.
"""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# Import your API routers here (example: from .api import router as api_router)
# from app.api import router as api_router

app = FastAPI(title="Hybrid RAG API", version="1.0.0")

# CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers here
# app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Hybrid RAG FastAPI is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
