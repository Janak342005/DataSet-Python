import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

CSV = "student_sleep_screen_time_academic_performance.csv"
COLS = ['daily_screen_time_hours', 'avg_sleep_hours_weekday', 'cgpa']

df = pd.read_csv(CSV)
print("Raw shape:", df.shape)
print("Missing values per column:\n", df[COLS].isna().sum())

# ---------- 1. Cleaning ----------
clean = df.dropna(subset=COLS).drop_duplicates().copy()
# remove implausible values (adjust to your data's CGPA scale)
clean = clean[(clean['daily_screen_time_hours'].between(0, 24)) &
              (clean['avg_sleep_hours_weekday'].between(0, 24))]
print("Clean shape:", clean.shape)

# ---------- 2. Descriptive statistics ----------
print(clean[COLS].describe().round(3))
print("\nSkewness:\n", clean[COLS].skew().round(3))

# ---------- 3. Correlations ----------
corr = clean[COLS].corr()
print("\nPearson correlation:\n", corr.round(3))
print("\nSpearman correlation:\n", clean[COLS].corr(method='spearman').round(3))
for c in COLS[:2]:
    r, p = stats.pearsonr(clean[c], clean['cgpa'])
    print(f"{c} vs cgpa: r={r:.3f}, p={p:.4g}")

# ---------- 4. Figures ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(ax, COLS):
    sns.histplot(clean[c], kde=True, ax=a)
    a.set_title(f"Distribution of {c}")
plt.tight_layout(); plt.savefig("fig1_distributions.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.regplot(data=clean, x='daily_screen_time_hours', y='cgpa', ax=ax[0],
            scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
ax[0].set_title("Screen Time vs CGPA")
sns.regplot(data=clean, x='avg_sleep_hours_weekday', y='cgpa', ax=ax[1],
            scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
ax[1].set_title("Weekday Sleep vs CGPA")
plt.tight_layout(); plt.savefig("fig2_scatter_fits.png", dpi=200); plt.close()

plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation matrix")
plt.tight_layout(); plt.savefig("fig3_corr.png", dpi=200); plt.close()

# ---------- 5. Models ----------
Y = clean['cgpa']

# Model A: multiple linear regression
XA = sm.add_constant(clean[COLS[:2]])
mA = sm.OLS(Y, XA).fit()
print("\n=== Model A: linear ===\n", mA.summary())

# Model B: with interaction term
cb = clean.copy()
cb['screen_x_sleep'] = cb['daily_screen_time_hours'] * cb['avg_sleep_hours_weekday']
XB = sm.add_constant(cb[COLS[:2] + ['screen_x_sleep']])
mB = sm.OLS(Y, XB).fit()
print("\n=== Model B: interaction ===\n", mB.summary())

# Model C: quadratic (diminishing / optimal-sleep effect)
cc = clean.copy()
cc['screen_sq'] = cc['daily_screen_time_hours'] ** 2
cc['sleep_sq'] = cc['avg_sleep_hours_weekday'] ** 2
XC = sm.add_constant(cc[COLS[:2] + ['screen_sq', 'sleep_sq']])
mC = sm.OLS(Y, XC).fit()
print("\n=== Model C: quadratic ===\n", mC.summary())

comp = pd.DataFrame({
    'Model': ['A linear', 'B interaction', 'C quadratic'],
    'R2': [mA.rsquared, mB.rsquared, mC.rsquared],
    'Adj_R2': [mA.rsquared_adj, mB.rsquared_adj, mC.rsquared_adj],
    'AIC': [mA.aic, mB.aic, mC.aic],
    'BIC': [mA.bic, mB.bic, mC.bic]}).round(4)
print("\nModel comparison:\n", comp)
comp.to_csv("model_comparison.csv", index=False)

# ---------- 6. Assumption checks (Model A) ----------
vif = pd.DataFrame({'variable': XA.columns,
                    'VIF': [variance_inflation_factor(XA.values, i) for i in range(XA.shape[1])]})
print("\nVIF:\n", vif.round(3))
bp = het_breuschpagan(mA.resid, XA)
print(f"Breusch-Pagan p-value: {bp[1]:.4f}")
sh = stats.shapiro(mA.resid.sample(min(len(mA.resid), 5000), random_state=1))
print(f"Shapiro-Wilk p-value (residuals): {sh[1]:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(mA.fittedvalues, mA.resid, alpha=0.4)
ax[0].axhline(0, color='red'); ax[0].set_title("Residuals vs Fitted")
sm.qqplot(mA.resid, line='s', ax=ax[1]); ax[1].set_title("Q-Q plot")
sns.histplot(mA.resid, kde=True, ax=ax[2]); ax[2].set_title("Residual distribution")
plt.tight_layout(); plt.savefig("fig4_diagnostics.png", dpi=200); plt.close()

# ---------- 7. Prediction surface ----------
s = np.linspace(clean['daily_screen_time_hours'].min(), clean['daily_screen_time_hours'].max(), 30)
h = np.linspace(clean['avg_sleep_hours_weekday'].min(), clean['avg_sleep_hours_weekday'].max(), 30)
S, H = np.meshgrid(s, h)
Z = mA.params['const'] + mA.params['daily_screen_time_hours'] * S + mA.params['avg_sleep_hours_weekday'] * H
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, H, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel("Screen time (h)"); ax.set_ylabel("Sleep (h)"); ax.set_zlabel("Predicted CGPA")
plt.tight_layout(); plt.savefig("fig5_surface.png", dpi=200); plt.close()

print("\nDone. Figures saved: fig1-fig5, model_comparison.csv")
