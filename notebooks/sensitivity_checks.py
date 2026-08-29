import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("../data/emdat_philippines_clean.csv")
df["log_damage"] = np.log(df["damage_000usd"])

# 1. Refit excluding Haiyan -- does the ICC change?
no_haiyan = df[df["DisNo."] != "2013-0433-PHL"]
m2 = smf.mixedlm("log_damage ~ storm_category", data=no_haiyan, groups=no_haiyan["province"]).fit(reml=True)
icc2 = m2.cov_re.iloc[0, 0] / (m2.cov_re.iloc[0, 0] + m2.scale)
print("--- Without Haiyan ---")
print(f"N={len(no_haiyan)}  Group Var={m2.cov_re.iloc[0,0]:.4f}  Scale={m2.scale:.4f}  ICC={icc2:.3f}")
print()

# 2. Raw (unconditional) between-province spread vs total spread, ignoring storm category
province_means = df.groupby("province")["log_damage"].mean()
print("--- Raw spread (no category control) ---")
print(f"Between-province variance of province means: {province_means.var():.3f}")
print(f"Overall variance of log_damage: {df['log_damage'].var():.3f}")
print()

# 3. How much does storm_category alone explain, with no province term at all?
X = sm.add_constant(df["storm_category"])
ols = sm.OLS(df["log_damage"], X).fit()
print("--- Plain OLS (no province) ---")
print(f"R-squared from storm_category alone: {ols.rsquared:.3f}")
print(f"Residual variance after storm_category: {ols.resid.var():.3f}")