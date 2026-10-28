import sys
import os

# Remove current project root or any path containing local wandb directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != project_root]

import wandb

print("Successfully imported real wandb library!")
print("WandB version:", wandb.__version__)

api = wandb.Api()
runs = api.runs("Craftax-50M-Benchmark")
print(f"Found {len(runs)} runs in Craftax-50M-Benchmark:")
for r in runs[:20]:
    print(f"Name: {r.name} | ID: {r.id} | State: {r.state} | Group: {r.group}")
