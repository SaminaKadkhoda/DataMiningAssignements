import json
from pathlib import Path
import matplotlib.pyplot as plt

log_dir = Path("sweep_logs")
runs = {
    "3e-4": Path("rate3e4.json"),
    "1e-3": Path("rate1e3.json"),
    "3e-3": Path("rate3e3.json"),
    "1e-2": Path("rate1e2.json"),
}

fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(9, 8))

for label, path in runs.items():
    with path.open(encoding="utf-8") as f:
        records = json.load(f)["steps"]

    train = [r for r in records if "loss" in r]
    valid = [r for r in records if "val_loss" in r]

    ax_train.plot(
        [r["step"] for r in train],
        [r["loss"] for r in train],
        label=f"LR {label}",
    )
    ax_val.plot(
        [r["step"] for r in valid],
        [r["val_loss"] for r in valid],
        marker="o",
        label=f"LR {label}",
    )

ax_train.set_title("Learning-rate sweep: training loss")
ax_train.set_xlabel("Training step")
ax_train.set_ylabel("Loss")
ax_train.set_xlim(0, 4883)
ax_train.set_xticks(range(0, 4884, 1000))
ax_train.grid(True)
ax_train.legend()

ax_val.set_title("Learning-rate sweep: validation loss")
ax_val.set_xlabel("Training step")
ax_val.set_ylabel("Validation loss")
ax_val.set_xlim(0, 4883)
ax_val.set_xticks(range(0, 4884, 1000))
ax_val.grid(True)
ax_val.legend()

fig.tight_layout()
fig.savefig("sweep_loss_curves.png", dpi=200, bbox_inches="tight")
plt.show()