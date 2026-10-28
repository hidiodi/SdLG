import glob
import os
import csv
import sys

# Ensure local wandb folder doesn't shadow anything
sys.path = [p for p in sys.path if os.path.abspath(p) != "/home/coins/PycharmProjects/JupyterProject"]

files = glob.glob("/home/coins/PycharmProjects/JupyterProject/checkpoints/craftax_50m_watchdog_T*/*_log.csv")
print(f"Total CSV log files found: {len(files)}")

summary = []
for f in sorted(files):
    dir_name = os.path.basename(os.path.dirname(f))
    file_name = os.path.basename(f)
    with open(f, "r") as fp:
        reader = list(csv.DictReader(fp))
        # Find last non-empty row
        last = None
        for row in reversed(reader):
            if row and any(v.strip() for v in row.values() if v):
                last = row
                break
        if last:
            summary.append({
                "dir": dir_name,
                "file": file_name,
                "arch": last.get("Architecture"),
                "seed": last.get("Seed"),
                "steps": last.get("Total_Steps"),
                "return": last.get("Train_Return"),
                "sps": last.get("SPS"),
                "vloss": last.get("Train_VLoss"),
                "entropy": last.get("Train_Entropy")
            })

print(f"{'Directory':<30} | {'File':<32} | {'Arch':<12} | {'Seed':<4} | {'Steps (M)':<9} | {'Return':<8} | {'SPS':<6}")
print("-" * 110)
for item in summary:
    steps_m = float(item['steps']) / 1e6 if item['steps'] else 0.0
    ret = float(item['return']) if item['return'] else 0.0
    sps = float(item['sps']) if item['sps'] else 0.0
    print(f"{item['dir']:<30} | {item['file']:<32} | {str(item['arch']):<12} | {str(item['seed']):<4} | {steps_m:<9.2f} | {ret:<8.3f} | {sps:<6.0f}")
