# Gunicorn configuration file for RFP Backend
# This file provides production-ready settings for multi-user deployment

import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', 8000)}"
backlog = 2048

# Worker processes
workers = 1  # Reduced to 1 since we have 4 background workers handling jobs
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 3000  # Increased timeout for long-running API calls
keepalive = 2

# Restart workers periodically to prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "rfp-backend"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
preload_app = True
reload = False
