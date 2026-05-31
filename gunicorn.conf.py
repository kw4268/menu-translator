# Settings the production server (gunicorn) loads automatically on startup.

# One worker keeps memory low on the free tier; threads let it handle several
# requests at once (e.g. the page fetching photos for many dishes in parallel).
workers = 1
threads = 8

# Give the AI time to read large menus. The default 30s was too short and caused
# big-menu requests to be killed mid-way (the app would appear to hang forever).
timeout = 120

# Recycle the worker occasionally to avoid any slow memory creep.
max_requests = 200
max_requests_jitter = 40
