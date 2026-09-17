import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))
    print(f"Starting ResearchLens API server at http://{host}:{port}")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)
