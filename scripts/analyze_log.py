import re
import sys

if len(sys.argv) != 2:
    print("Usage: python analyze_log.py path/to/FL.log")
    sys.exit(1)

file = sys.argv[1]

with open(file, "r", errors="ignore") as f:
    text = f.read()

pattern = re.compile(
    r'\[Round:\s*(\d+)\].*?Evaluate global model\'s performance\.\.\!.*?'
    r'=> Loss:\s*([0-9.]+).*?'
    r'=> Accuracy:\s*([0-9.]+)%.*?'
    r'=> Precision:\s*([0-9.]+).*?'
    r'=> Recall:\s*([0-9.]+).*?'
    r'=> F1:\s*([0-9.]+).*?'
    r'=> Round time:\s*([0-9.]+)\s*sec',
    re.S
)

rows = []

for m in pattern.finditer(text):
    rows.append({
        "round": int(m.group(1)),
        "loss": float(m.group(2)),
        "accuracy": float(m.group(3)),
        "precision": float(m.group(4)),
        "recall": float(m.group(5)),
        "f1": float(m.group(6)),
        "time": float(m.group(7))
    })

if not rows:
    print("No complete evaluated rounds found.")
    sys.exit(1)

best_acc = max(rows, key=lambda x: x["accuracy"])
best_precision = max(rows, key=lambda x: x["precision"])
best_recall = max(rows, key=lambda x: x["recall"])
best_f1 = max(rows, key=lambda x: x["f1"])
best_loss = min(rows, key=lambda x: x["loss"])

final = max(rows, key=lambda x: x["round"])

total_time = sum(x["time"] for x in rows)
avg_time = total_time / len(rows)

print()
print("=" * 65)
print("LOG:", file)
print("=" * 65)

print(f"\nEvaluated rounds: {len(rows)}")

print("\n===== BEST RESULTS =====")
print(f"Best Accuracy : {best_acc['accuracy']:.2f}% (Round {best_acc['round']})")
print(f"Best Precision: {best_precision['precision']:.4f} (Round {best_precision['round']})")
print(f"Best Recall   : {best_recall['recall']:.4f} (Round {best_recall['round']})")
print(f"Best F1       : {best_f1['f1']:.4f} (Round {best_f1['round']})")
print(f"Lowest Loss   : {best_loss['loss']:.4f} (Round {best_loss['round']})")

print("\n===== FINAL ROUND =====")
print(f"Round         : {final['round']}")
print(f"Accuracy      : {final['accuracy']:.2f}%")
print(f"Precision     : {final['precision']:.4f}")
print(f"Recall        : {final['recall']:.4f}")
print(f"F1            : {final['f1']:.4f}")
print(f"Loss          : {final['loss']:.4f}")

print("\n===== TRAINING TIME =====")
print(f"Total seconds : {total_time:.2f}")
print(f"Total minutes : {total_time/60:.2f}")
print(f"Total hours   : {total_time/3600:.2f}")
print(f"Avg/round     : {avg_time:.2f} sec")

print("=" * 65)
