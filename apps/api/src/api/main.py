from fastapi import FastAPI

app = FastAPI(title="ResumeFit API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
