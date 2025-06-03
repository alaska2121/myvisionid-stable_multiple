from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.middleware.rate_limiter import rate_limit_middleware
from app.middleware.error_handler import error_handler_middleware
from app.middleware.request_logger import request_logger_middleware
from app.middleware.timeout import timeout_middleware

app = FastAPI(
    title="Background Change API",
    description="API for removing backgrounds from photos",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(timeout_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(error_handler_middleware)
app.middleware("http")(request_logger_middleware)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 