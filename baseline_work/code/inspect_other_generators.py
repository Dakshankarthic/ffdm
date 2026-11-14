import requests

generators = ['sd2', 'sdxl', 'sd3', 'flux']
for gen in generators:
    print(f"=== Inspecting generator: {gen} ===", flush=True)
    for off in [0, 5000, 10000, 15000]:
        p = {'dataset': 'nebula/OpenSDI_test', 'config': 'default', 'split': gen, 'offset': off, 'length': 1}
        try:
            r = requests.get('https://datasets-server.huggingface.co/rows', params=p, timeout=20)
            if r.status_code == 200:
                rows = r.json().get('rows', [])
                if rows:
                    row = rows[0]['row']
                    has_mask = (row.get('mask') is not None) and (row['mask'].get('src') is not None)
                    print(f"  {gen} offset {off}: key={row.get('key')}, label={row.get('label')}, has_mask={has_mask}", flush=True)
            else:
                print(f"  {gen} offset {off}: status {r.status_code}", flush=True)
        except Exception as e:
            print(f"  {gen} offset {off}: error {e}", flush=True)

