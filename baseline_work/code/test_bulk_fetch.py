import requests

p = {'dataset': 'nebula/OpenSDI_test', 'config': 'default', 'split': 'sd15', 'offset': 10000, 'length': 10}
r = requests.get('https://datasets-server.huggingface.co/rows', params=p, timeout=30)
print('Status:', r.status_code)
if r.status_code == 200:
    rows = r.json().get('rows', [])
    print(f'Got {len(rows)} rows in 1 single request!')
    for i, row in enumerate(rows):
        r_item = row['row']
        has_mask = bool(r_item.get('mask') and r_item['mask'].get('src'))
        print(f"  {i}: key={r_item['key']}, label={r_item['label']}, has_mask={has_mask}")

