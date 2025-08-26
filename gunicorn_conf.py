# gunicorn_conf.py
# Use the uvicorn worker class, which is designed for async applications like FastAPI.
workers = 1  # For the free tier, a single worker is sufficient and recommended.
worker_class = "uvicorn.workers.UvicornWorker"