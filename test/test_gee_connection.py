from sar_lulc_monitor.network import enable_ipv4_only

enable_ipv4_only()

import ee

print("Initializing Earth Engine...", flush=True)

ee.Initialize(project="deep-learning-exam-2026")

print("Initialization successful!", flush=True)

print("GEE response:", ee.Number(1).getInfo(), flush=True)
