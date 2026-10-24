"""Fichier CSV d'import Metricool pour un lot (calendrier > Import CSV).

    python3 -m generator.metricool_csv --date 2026-10-25 --times 08:05,09:35,...
Écrit /mnt/project-files/metricool/<date>.csv. Colonnes d'après l'aide Metricool ; comparer avec
« Download template » en cas de refus à l'import.
"""
import argparse
import csv
import os

from .batch import load_log

TIMES = ["08:05", "09:35", "10:50", "12:20", "13:40", "14:55", "16:05", "17:15", "18:25", "19:40"]
NETWORKS = ["Facebook", "Twitter", "LinkedIn", "GBP", "Instagram", "Pinterest", "TikTok", "YouTube", "Threads", "Bluesky"]
COLS = (["Text", "Date", "Time", "Draft"] + NETWORKS + ["Picture Url 1", "First Comment Text", "YouTube Video Title",
        "YouTube Video Type", "YouTube Video Privacy", "YouTube video for kids", "YouTube Video Category"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--times", default=",".join(TIMES))
    ap.add_argument("--out", default="/mnt/project-files/metricool/")
    args = ap.parse_args()
    times = args.times.split(",")
    entries = [e for e in load_log() if e["created"] == args.date and e["status"] == "rendered"]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.date}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for t, e in zip(times, entries):
            row = {n: "FALSE" for n in NETWORKS}
            row.update({"Text": e["description"], "Date": args.date, "Time": t + ":00", "Draft": "FALSE", "YouTube": "TRUE",
                        "Picture Url 1": e["url"], "First Comment Text": e["first_comment"], "YouTube Video Title": e["title"],
                        "YouTube Video Type": "SHORT", "YouTube Video Privacy": "PUBLIC", "YouTube video for kids": "false",
                        "YouTube Video Category": "GAMING"})
            w.writerow(row)
    print(path, len(entries))


if __name__ == "__main__":
    main()
