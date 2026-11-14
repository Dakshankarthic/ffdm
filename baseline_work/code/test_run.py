import sys
import traceback
from pathlib import Path
sys.path.insert(0, "I:/icvgip workshop/baseline_work/code")

try:
    print("Trying import baseline_adapter...", flush=True)
    import baseline_adapter
    print("Success baseline_adapter!", flush=True)
    print("Trying import test_metrics_and_checks...", flush=True)
    import test_metrics_and_checks
    print("Success test_metrics_and_checks!", flush=True)
except BaseException as e:
    print(f"Caught exception: {type(e)}: {e}", flush=True)
    traceback.print_exc()
