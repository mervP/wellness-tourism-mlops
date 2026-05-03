"""Register the raw tourism dataset on the Hugging Face Hub.

Idempotent: skips the upload when ``tourism.csv`` already exists on the Hub.
"""

import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


HF_USERNAME = os.getenv("HF_USERNAME", "prashanth-merwyn")
REPO_ID = f"{HF_USERNAME}/wellness-tourism-dataset"
REPO_TYPE = "dataset"
DATA_FILE = "tourism.csv"
LOCAL_FOLDER = "tourism_project/data"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists. Reusing it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{REPO_ID}' not found. Creating new repo...")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False,
                token=os.getenv("HF_TOKEN"))

existing_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
if DATA_FILE in existing_files:
    print(f"'{DATA_FILE}' is already on the Hub - skipping upload.")
else:
    local_path = os.path.join(LOCAL_FOLDER, DATA_FILE)
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"'{DATA_FILE}' not found on the Hub and no local copy at {local_path}."
        )
    api.upload_file(path_or_fileobj=local_path, path_in_repo=DATA_FILE,
                    repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Uploaded {DATA_FILE} to {REPO_ID}.")
