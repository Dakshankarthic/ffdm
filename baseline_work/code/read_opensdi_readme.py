import requests

r = requests.get("https://huggingface.co/datasets/nebula/OpenSDI_test/raw/main/README.md")
print("=== README of nebula/OpenSDI_test ===")
print(r.text[:3000])

