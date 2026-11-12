"""Print results/results.csv as a table."""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "results/results.csv")))
print(f"{'metric':26s} {'target':7s} {'measured':12s} pass")
for r in rows:
    m = f"{r['measured']} {r['unit']}" if r["measured"] else "(phone run)"
    print(f"{r['metric']:26s} {r['target']:7s} {m:12s} {r['pass'] or '-'}")
