"""Register the raw tourism dataset on the Hugging Face Hub."""

import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


HF_USERNAME = os.getenv("HF_USERNAME", "prashanth-merwyn")
REPO_ID = f"{HF_USERNAME}/wellness-tourism-dataset"
REPO_TYPE = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists. Reusing it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{REPO_ID}' not found. Creating new repo...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Dataset repo '{REPO_ID}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print("Raw dataset uploaded to the Hugging Face Hub.")
