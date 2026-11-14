import requests

r = requests.get("https://huggingface.co/api/datasets/nebula/OpenSDI_test")
data = r.json()
siblings = [s["rfilename"] for s in data.get("siblings", [])]
print("All files in nebula/OpenSDI_test:")
for s in siblings:
    print(" ", s)

