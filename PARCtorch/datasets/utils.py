import os
import requests
import zipfile
from io import BytesIO
from tqdm import tqdm
from pathlib import Path


def download(url, target_path, chunk_size=1024):
    """
    Downloads a file from the specified URL and saves it to the target_path.
    """
    print(f"Downloading a file from {url}")
    if os.path.exists(target_path):
        print(f"Found a cached data in {target_path}. Aborting the download.")
        print(
            "If you want to re-download the data, please remove the cached data and try again."
        )
        return target_path

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    progress = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading to {target_path}")

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()

    print(f"Download complete. File saved to {target_path}")
    return target_path


def extract_zip(zip_path, target_folder):
    """
    Extracts the contents of a ZIP file at zip_path into the target folder.
    """
    Path(target_folder).mkdir(parents=True, exist_ok=True)
    print(f"Extracting contents to {target_folder} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_folder)
    print("Extraction complete.")
