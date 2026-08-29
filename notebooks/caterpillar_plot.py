import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = pd.read_csv("../data/emdat_philippines_clean.csv")
df["log_damage"] = np.log(df["damage_000usd"])

model = smf.mixedlm("log_damage ~ storm_category", data=df, groups=df["province"])
result = model.fit(reml=True)

re = result.random_effects
re_cov = result.random_effects_cov
re_df = pd.DataFrame({
    "province": list(re.keys()),
    "intercept_re": [float(v.iloc[0]) if hasattr(v, "iloc") else float(v[0]) for v in re.values()],
    "se": [float(np.sqrt(re_cov[p].iloc[0, 0])) for p in re.keys()],
})
re_df["ci_lo"] = re_df["intercept_re"] - 1.96 * re_df["se"]
re_df["ci_hi"] = re_df["intercept_re"] + 1.96 * re_df["se"]
re_df = re_df.sort_values("intercept_re").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 16))
colors = ["#c0392b" if (lo > 0 or hi < 0) else "#7f8c8d" for lo, hi in zip(re_df["ci_lo"], re_df["ci_hi"])]
for i, row in re_df.iterrows():
    ax.errorbar(row["intercept_re"], i,
                xerr=[[row["intercept_re"] - row["ci_lo"]], [row["ci_hi"] - row["intercept_re"]]],
                fmt="o", color=colors[i], capsize=2, markersize=4, elinewidth=1)
ax.axvline(0, color="gray", linestyle="--", linewidth=1)
ax.set_yticks(range(len(re_df)))
ax.set_yticklabels(re_df["province"], fontsize=7)
ax.set_xlabel("Random intercept (log-damage scale)")
ax.set_title("Province-level random effects\n(after accounting for storm category) -- red = 95% CI excludes zero")
plt.tight_layout()
plt.savefig("caterpillar_plot.png", dpi=150)
print("Saved caterpillar_plot.png")

n_sig = sum(1 for lo, hi in zip(re_df.ci_lo, re_df.ci_hi) if lo > 0 or hi < 0)
print(f"Provinces with 95% CI excluding zero: {n_sig}")