from fastapi import Request
import time
import logging

async def request_logger_middleware(request: Request, call_next):
    start_time = time.time()
    logging.info(f"Request started: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logging.info(
        f"Request completed: {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Time: {process_time:.2f}s"
    )
    
    return response 