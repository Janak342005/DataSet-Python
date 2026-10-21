import pandas as pd, numpy as np, matplotlib.pyplot as plt

d = pd.read_csv("student_sleep_screen_time_academic_performance.csv").dropna(
    subset=["daily_screen_time_hours", "avg_sleep_hours_weekday", "cgpa"]
)

d["b"] = pd.cut(
    d.avg_sleep_hours_weekday,
    [0, 5, 6, 7, 8, 12],
    labels=["≤5", "5–6", "6–7", "7–8", ">8"],
)

g = d.groupby("b", observed=True).cgpa.agg(["mean", "std", "count"])

g["se"] = g["std"] / np.sqrt(g["count"])

plt.figure(figsize=(6, 4))

plt.bar(g.index.astype(str), g["mean"], yerr=1.96 * g["se"], capsize=4, color="#4C78A8")

for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
    plt.text(i, 2.0, f"n={n}", ha="center", color="white", fontsize=9)
    plt.ylim(1.9, 3.3)
    plt.xlabel("Weekday sleep (hours)")
    plt.ylabel("Mean CGPA (95% CI)")
    plt.title("Mean CGPA by sleep band")
    plt.tight_layout()
    plt.savefig("fig6_sleep_bands.png", dpi=200)
