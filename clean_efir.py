"""Quality control of the Efir 1.3 one-minute record (October 2019).
Flagged values are set to NaN (missing); no rows are removed, so the time grid is unchanged.
R1  CO and SO2 simultaneously equal to 0  -> CO, SO2 = NaN (QC assumption: probable instrument downtime encoded as zero; no operation logs available)
R2  malfunction episode 2019-10-01 19:40 - 2019-10-02 01:06 -> all pollutant channels = NaN
R3  any pollutant value >= 30000 ug/m3 (sensor limit code) -> NaN
R4  relative humidity > 100 % (error code 3276.7) -> NaN
Usage: python clean_efir.py [input.xlsx] [output.xlsx]
Defaults: 2019-10-efir_1-3.xlsx -> 2019-10-efir_1-3_clean.xlsx (in the script folder)
"""
import sys
from pathlib import Path
import pandas as pd

here = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / '2019-10-efir_1-3.xlsx'
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else here / '2019-10-efir_1-3_clean.xlsx'
print(f'input:  {src}\noutput: {dst}')
df = pd.read_excel(src)
tcol = df.columns[0]
t = pd.to_datetime(df[tcol], dayfirst=True)
num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', '.'), errors='coerce')
pol = [c for c in df.columns if 'мкг' in c]
co, so2, rh = 'CO (мкг/м3)', 'SO2 (мкг/м3)', 'RH (%)'
for c in df.columns[1:]:
    df[c] = num(df[c])
log = {}
m1 = (df[co] == 0) & (df[so2] == 0)
df.loc[m1, [co, so2]] = float('nan'); log['R1 joint zero CO&SO2'] = int(m1.sum())
m2 = (t >= '2019-10-01 19:40') & (t <= '2019-10-02 01:06')
df.loc[m2, pol] = float('nan'); log['R2 malfunction episode rows'] = int(m2.sum())
n3 = 0
for c in pol:
    m3 = df[c] >= 30000; n3 += int(m3.sum()); df.loc[m3, c] = float('nan')
log['R3 >=30000 (after R2)'] = n3
m4 = df[rh] > 100
df.loc[m4, rh] = float('nan'); log['R4 RH>100'] = int(m4.sum())
df.to_excel(dst, index=False)
for k, v in log.items():
    print(f'{k}: {v}')
for c in [co, so2, rh]:
    print(f'{c}: missing {df[c].isna().mean()*100:.1f}%  max {df[c].max():.1f}')
