import logging
import os
import httpx
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import uvicorn

logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "9000")
class Backend(BaseModel):
    url: str
    capacity: int
    current_load: Optional[int] = 0

def load_backends_from_config() -> List[Backend]:
    config_path = os.getenv("EXTERNAL_RUNNERS_CONFIG", "/config/external_runners_config.json")
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            return [Backend(**backend) for backend in config_data]
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except ValidationError as e:
        raise ValueError(f"Invalid config data: {e}")
    
def config_logging():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_logging()

    # Load backends during startup
    global backends
    backends = load_backends_from_config()
    print(f"Loaded backends: {backends}")

    logger.info(f"Started external container manager")
    yield
    logger.info("Shutting down")
     # Function to choose the backend based on available capacity

# FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

def choose_backend():
    available_backends = [b for b in backends if b.current_load < b.capacity]
    if not available_backends:
        raise HTTPException(status_code=500, detail="All backends are at capacity.")
    backend = min(available_backends, key=lambda b: b.current_load)
    return backend

# Route that proxies the request to the chosen backend, including dynamic path
@app.api_route("/{full_path:path}", methods=["POST"])
async def proxy_request(request: Request, full_path: str):
    backend = choose_backend()
    logger.info(f"Proxying request to backend: {backend.url}")
    # Update the backend's current load
    backend.current_load += 1

    # Build the full proxied URL by appending the incoming path
    proxied_url = f"{backend.url}/{full_path}"
    headers = dict(request.headers)
    headers["host"] = backend.url
    async with httpx.AsyncClient() as client:
        client.timeout = None #disable timeout
        try:
            # Forward the request method, headers, and body to the proxied URL
            response = await client.request(
                method=request.method,
                url=proxied_url,
                headers=headers,
                content=await request.body(),
            )

            # Return the response from the proxied request
            return response.json()

        except Exception as e:
            logger.exception(f"Failed to proxy request: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to proxy request: {e}")

        finally:
            # Decrease load after request is complete
            backend.current_load -= 1

if __name__ == "__main__":
    #run the api
    uvicorn.run(
        "external_runner_manager:app",
        host=HOST,
        port=int(PORT)
    )
