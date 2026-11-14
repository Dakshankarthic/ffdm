import requests

def inspect_keys(dataset, split, offsets):
    print(f"=== Inspecting {dataset} split={split} ===")
    for off in offsets:
        p = {'dataset': dataset, 'config': 'default', 'split': split, 'offset': off, 'length': 1}
        r = requests.get('https://datasets-server.huggingface.co/rows', params=p, timeout=20)
        if r.status_code == 200:
            rows = r.json().get('rows', [])
            if rows:
                row = rows[0]['row']
                has_mask = (row.get('mask') is not None) and (row['mask'].get('src') is not None)
                print(f"Offset {off}: key={row.get('key')}, label={row.get('label')}, has_mask={has_mask}")
        else:
            print(f"Offset {off}: error {r.status_code}")

inspect_keys('nebula/OpenSDI_test', 'sd15', [0, 500, 1000, 2000, 5000, 10000, 12000, 15000, 18000])

