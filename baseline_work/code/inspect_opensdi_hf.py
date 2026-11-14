import requests
import json

def inspect_dataset(repo_id):
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    r = requests.get(url, timeout=15)
    print(f"=== Dataset API: {repo_id} (Status: {r.status_code}) ===")
    if r.status_code == 200:
        data = r.json()
        print("ID:", data.get("id"))
        print("Description:", (data.get("description") or "")[:200])
        print("Tags:", data.get("tags"))
        print("Private:", data.get("private"))
        print("Gated:", data.get("gated"))
        siblings = [s["rfilename"] for s in data.get("siblings", [])]
        print("Files count:", len(siblings))
        print("First 15 files:", siblings[:15])

print("Testing OpenSDI datasets on HF...")
inspect_dataset("nebula/OpenSDI_test")
inspect_dataset("nebula/OpenSDI_train")

