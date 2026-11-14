import sys
from pathlib import Path
sys.path.insert(0, "I:/icvgip workshop/baseline_work/code")

try:
    print("Testing import baseline_adapter...")
    import baseline_adapter
    print("baseline_adapter imported successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    print("Testing import test_metrics_and_checks...")
    import test_metrics_and_checks
    print("test_metrics_and_checks imported successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    print("Testing import cv2...")
    import cv2
    print("cv2 imported successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()

