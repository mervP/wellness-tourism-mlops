"""Push the deployment artefacts to the Hugging Face Streamlit Space."""

import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


HF_USERNAME = os.getenv("HF_USERNAME", "prashanth-merwyn")
SPACE_REPO = f"{HF_USERNAME}/wellness-tourism-prediction"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists. Reusing it.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO}' not found. Creating Streamlit Docker space...")
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        private=False,
        token=os.getenv("HF_TOKEN"),
    )

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment artefacts pushed to https://huggingface.co/spaces/{SPACE_REPO}")
