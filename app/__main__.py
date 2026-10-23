import uvicorn

from .config import Settings

if __name__ == "__main__":
    s = Settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port)
