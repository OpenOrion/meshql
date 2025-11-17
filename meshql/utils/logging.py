import logging

# Configure the logger to log to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger("meshql")
