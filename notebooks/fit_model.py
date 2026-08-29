import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("../data/emdat_philippines_clean.csv")
df["log_damage"] = np.log(df["damage_000usd"])

model = smf.mixedlm(
    "log_damage ~ storm_category",
    data=df,
    groups=df["province"],
)
result = model.fit(reml=True)
print(result.summary())

icc = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale)
print(f"\nICC (share of residual variance from province): {icc:.3f}")