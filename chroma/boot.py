import os
import subprocess
from google.cloud import storage
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("boot")

def download_bucket(bucket_name, destination_folder):
    """Downloads a GCS bucket to a local folder."""
    logger.info(f"Connecting to GCS bucket: {bucket_name}")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs())
        
        if not blobs:
            logger.warning(f"Bucket {bucket_name} appears to be empty.")
            return

        logger.info(f"Found {len(blobs)} files to download.")
        
        os.makedirs(destination_folder, exist_ok=True)
        
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
                
            local_path = os.path.join(destination_folder, blob.name)
            local_dir = os.path.dirname(local_path)
            
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
                
            logger.info(f"Downloading {blob.name} -> {local_path}")
            blob.download_to_filename(local_path)
            
        logger.info("Download complete.")
        
    except Exception as e:
        logger.error(f"Failed to download bucket: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python boot.py <bucket_name>")
        sys.exit(1)
        
    bucket_name = sys.argv[1]
    data_dir = "/data"
    
    # 1. Download Data
    download_bucket(bucket_name, data_dir)
    
    # 2. Start Chroma
    logger.info("Starting ChromaDB...")
    cmd = ["chroma", "run", "--path", data_dir, "--host", "0.0.0.0", "--port", "8000"]
    
    # Replace current process with Chroma
    os.execvp("chroma", cmd)
