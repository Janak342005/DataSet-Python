import pandas as pd, numpy as np, statsmodels.api as sm

df = pd.read_csv("student_sleep_screen_time_academic_performance.csv")

c = ["daily_screen_time_hours", "avg_sleep_hours_weekday", "cgpa"]

d = df.dropna(subset=c)

print(d.isna().sum()[d.isna().sum() > 0])

print(
    d[
        [
            "pre_sleep_screen_time_min",
            "sleep_quality_score",
            "weekly_study_hours",
            "class_attendance_percent",
            "concentration_difficulty",
            "energy_level",
        ]
    ]
    .describe()
    .round(2)
)

print(
    d[
        [
            "daily_screen_time_hours",
            "avg_sleep_hours_weekday",
            "pre_sleep_screen_time_min",
            "sleep_quality_score",
            "weekly_study_hours",
            "class_attendance_percent",
            "concentration_difficulty",
            "energy_level",
            "cgpa",
        ]
    ]
    .corr()["cgpa"]
    .round(3)
)

print(d.groupby("device_free_before_bed").cgpa.agg(["mean", "count"]).round(3))

print(
    "screen>14:",
    (d.daily_screen_time_hours > 14).sum(),
    d[d.daily_screen_time_hours > 14][c].values.tolist(),
)

o = d[d.daily_screen_time_hours <= 14]

m = sm.OLS(o.cgpa, sm.add_constant(o[c[:2]])).fit()
print(
    "no-outlier",
    m.params.round(4).to_dict(),
    m.pvalues.round(4).to_dict(),
    round(m.rsquared, 4),
    len(o),
)

# Model D
dd = d.dropna(
    subset=[
        "sleep_quality_score",
        "weekly_study_hours",
        "class_attendance_percent",
        "pre_sleep_screen_time_min",
    ]
).copy()

X = sm.add_constant(
    dd[
        [
            "daily_screen_time_hours",
            "avg_sleep_hours_weekday",
            "pre_sleep_screen_time_min",
            "weekly_study_hours",
            "class_attendance_percent",
            "sleep_quality_score",
        ]
    ]
)

mD = sm.OLS(dd.cgpa, X).fit()
print(
    mD.summary().tables[1],
    round(mD.rsquared, 4),
    round(mD.rsquared_adj, 4),
    mD.aic,
    len(dd),
)

# Threshold / piecewise sleep
d = d.copy()
d["bins"] = pd.cut(d.avg_sleep_hours_weekday, [0, 5, 6, 7, 8, 12])

print(d.groupby("bins", observed=True).cgpa.agg(["mean", "std", "count"]).round(3))

# cgpa max
print(d.cgpa.min(), d.cgpa.max())
