# -*- coding: utf-8 -*-
"""
BR-CPA-RNet — single-file reproducible experiment
==================================================

This file is a single-file distribution of the manuscript-aligned
BR-CPA-RNet experiment. It embeds the two internal implementation modules
used by the final confirmatory experiment so that the GitHub repository can
contain one Python source file plus a README.

Primary manuscript setting:
    30-min resolution
    2-h history (T=4)
    1-h forecast horizon (H=2)
    fixed bounded cross-pollutant residual coefficient alpha=0.30
    compact Sentinel-5P context
    maximum satellite age 48 h
    3 rolling-origin folds
    5 neural initializations per fold
    paired moving-block bootstrap

Required user data:
    2019-10-efir_1-3.xlsx
    results_s5p_exact/s5p_aligned_30min.csv

This file does not contain credentials or raw monitoring data.
"""

from __future__ import annotations

import sys
import types


# -----------------------------------------------------------------------------
# Embedded module: air_quality_final_compact_cpa
# -----------------------------------------------------------------------------
_BASE_SOURCE = '# -*- coding: utf-8 -*-\n"""\nFINAL GROUND+METEOROLOGY CPA EXPERIMENT\n=======================================\n\nTemporal formulation is FROZEN from validation screening:\n    resampling = 30 min\n    history = 2 h  -> T = 4\n    forecast horizon = +1 h -> H = 2 steps\n    target = POINT\n\nScientifically usable targets in the current dataset:\n    CO and SO2\n\nModels\n------\n1. Persistence\n2. Previous-hour mean\n3. HistGradientBoosting residual baseline\n4. Residual LSTM\n5. Compact Residual CPA-BiLSTM\n6. Ablations:\n   - without CPA\n   - UniLSTM instead of BiLSTM\n   - without temporal attention\n   - without meteorological regime/anomaly context\n\nThe network predicts a correction to Persistence:\n    y_hat(t+H) = y_last(t) + lambda_p * delta_hat(t+H)\n\nlambda_p is selected ONLY on validation data.\n\nWhy "Compact"?\n--------------\nAt 30-min resolution and 2 h history, T=4. The previously proposed\nthree-layer dilated Conv1D block with dilation {1,2,4} has a receptive\nfield much larger than the input sequence, so it is not scientifically\nwell matched to the selected temporal formulation. Here each pollutant\nuses a compact local temporal encoder, while Cross-Pollutant Attention\nremains the key inter-pollutant module.\n\nNo TEST-driven tuning is performed in this script.\n"""\n\nfrom __future__ import annotations\n\nimport json\nimport math\nimport os\nimport random\nimport tempfile\nimport time\nimport warnings\nfrom copy import deepcopy\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Dict, List, Tuple\n\nwarnings.filterwarnings("ignore")\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom scipy import stats\n\nfrom sklearn.cluster import KMeans\nfrom sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor\nfrom sklearn.metrics import (\n    mean_absolute_error,\n    mean_squared_error,\n    r2_score,\n    precision_score,\n    recall_score,\n    f1_score,\n)\nfrom sklearn.preprocessing import RobustScaler\n\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import Dataset, DataLoader\n\n\n# ================================================================\n# 0. CONFIG\n# ================================================================\n\nDATA_PATH = Path("2019-10-efir_1-3.xlsx")\nOUT_DIR = Path("results_final_cpa")\nFIG_DIR = OUT_DIR / "figures"\nOUT_DIR.mkdir(exist_ok=True)\nFIG_DIR.mkdir(exist_ok=True)\n\nSEED = 42\n\nRESAMPLE_MINUTES = 30\nHISTORY_HOURS = 2\nHISTORY_STEPS = int(HISTORY_HOURS * 60 / RESAMPLE_MINUTES)  # 4\nFORECAST_HOURS = 1\nHORIZON_STEPS = int(FORECAST_HOURS * 60 / RESAMPLE_MINUTES)  # 2\n\nTRAIN_FRAC = 0.70\nVAL_FRAC = 0.15\n\nTARGETS = ["CO", "SO2"]\n\nBATCH_SIZE = 64\nMAX_EPOCHS = 60\nPATIENCE = 10\nLR = 8e-4\nWEIGHT_DECAY = 1e-5\n\n# Final paper should use >=5 seeds. With T=4 this should still be manageable on CPU.\nN_SEEDS = 5\nN_BOOTSTRAP = 5000\n\nLAMBDA_GRID = np.linspace(0.0, 1.0, 101)\n\nN_WEATHER_REGIMES = 4\nEVENT_QUANTILE = 0.90\n\nDEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n\nPOLLUTANT_ALIASES = {\n    "CO": ["CO (мкг/м3)", "CO (мкг/м³)", "CO"],\n    "NO2": ["NO2 (мкг/м3)", "NO₂ (мкг/м3)", "NO2", "NO₂"],\n    "SO2": ["SO2 (мкг/м3)", "SO₂ (мкг/м3)", "SO2", "SO₂"],\n    "O3": ["O3 (мкг/м3)", "O₃ (мкг/м3)", "O3", "O₃"],\n    "PM": ["Пил (мкг/м3)", "Пил (мкг/м³)", "Dust", "PM2.5", "PM10", "PM", "TSP"],\n}\n\nMETEO_ALIASES = {\n    "WS": ["WS (м/с)", "WS", "Wind speed"],\n    "WD": ["WD (град.)", "WD", "Wind direction"],\n    "Temperature": ["T (°C)", "Temperature", "Температура"],\n    "RH": ["RH (%)", "RH", "Relative humidity"],\n    "Pressure": ["Атм._тиск (гПа)", "Pressure", "Тиск"],\n}\n\nTIME_ALIASES = ["Дата/Час", "time", "Time", "timestamp", "datetime", "Дата"]\n\n\n# ================================================================\n# 1. REPRODUCIBILITY\n# ================================================================\n\ndef set_seed(seed=SEED):\n    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(seed)\n\nset_seed(SEED)\n\n\n# ================================================================\n# 2. LOADING / RESAMPLING\n# ================================================================\n\ndef norm(x):\n    return str(x).strip().replace("³","3").replace("₂","2").replace("₃","3").lower()\n\ndef find_col(columns, aliases):\n    d = {norm(c):c for c in columns}\n    for a in aliases:\n        if norm(a) in d:\n            return d[norm(a)]\n    for a in aliases:\n        na=norm(a)\n        for nc,c in d.items():\n            if na in nc or nc in na:\n                return c\n    return None\n\ndef to_num(s):\n    x=(s.astype("string")\n       .str.replace(",",".",regex=False)\n       .str.replace("\\u00a0","",regex=False)\n       .str.strip())\n    x=x.mask(x.isin(["","nan","None","<NA>"]))\n    return pd.to_numeric(x,errors="coerce")\n\ndef circ_mean(s):\n    x=pd.to_numeric(s,errors="coerce").dropna().to_numpy(float)\n    if len(x)==0:return np.nan\n    r=np.deg2rad(x)\n    a=np.arctan2(np.mean(np.sin(r)),np.mean(np.cos(r)))\n    return float((np.rad2deg(a)+360)%360)\n\ndef load_data():\n    df=pd.read_excel(DATA_PATH)\n    df.columns=[str(c).strip() for c in df.columns]\n\n    tc=find_col(df.columns,TIME_ALIASES) or df.columns[0]\n    df[tc]=pd.to_datetime(df[tc],errors="coerce",dayfirst=True)\n    df=df.dropna(subset=[tc]).sort_values(tc)\n\n    pcols={}\n    for p,a in POLLUTANT_ALIASES.items():\n        c=find_col(df.columns,a)\n        if c:\n            pcols[p]=c\n            df[c]=to_num(df[c])\n\n    mcols={}\n    for m,a in METEO_ALIASES.items():\n        c=find_col(df.columns,a)\n        if c:\n            mcols[m]=c\n            df[c]=to_num(df[c])\n\n    x=df.set_index(tc)\n    rule=f"{RESAMPLE_MINUTES}min"\n    out=pd.DataFrame(index=x.resample(rule).size().index)\n\n    for p,c in pcols.items():\n        out[c]=x[c].resample(rule).mean()\n\n    for m,c in mcols.items():\n        if m=="WD":\n            out[c]=x[c].resample(rule).apply(circ_mean)\n        else:\n            out[c]=x[c].resample(rule).mean()\n\n    missing=[p for p in TARGETS if p not in pcols]\n    if missing:\n        raise ValueError(f"Missing required target columns: {missing}")\n\n    return out,pcols,mcols\n\n\n# ================================================================\n# 3. CHRONOLOGICAL SPLIT\n# ================================================================\n\ndef split_indices(n):\n    i1=int(n*TRAIN_FRAC)\n    i2=int(n*(TRAIN_FRAC+VAL_FRAC))\n    return np.arange(i1),np.arange(i1,i2),np.arange(i2,n)\n\n\n# ================================================================\n# 4. FEATURE ENGINEERING\n# ================================================================\n\ndef time_since(mask):\n    out=[]\n    gap=0\n    for v in mask.to_numpy(bool):\n        if v: gap=0\n        else: gap+=1\n        out.append(gap*RESAMPLE_MINUTES/60)\n    return np.asarray(out,float)\n\ndef robust_spike_flag(s,window=24,zthr=8.0):\n    prev=s.shift(1)\n    med=prev.rolling(window,min_periods=8).median()\n    mad=(prev-med).abs().rolling(window,min_periods=8).median()\n    scale=(1.4826*mad).replace(0,np.nan)\n    z=(s-med).abs()/scale\n    return (z>zthr).fillna(False).astype(float)\n\ndef build_base_features(grid,pcols,mcols):\n    f=pd.DataFrame(index=grid.index)\n    branch={}\n\n    for p in TARGETS:\n        raw=grid[pcols[p]]\n        mask=raw.notna().astype(float)\n\n        f[f"{p}_value"]=raw.ffill()\n        f[f"{p}_mask"]=mask\n        f[f"{p}_age_h"]=time_since(mask)\n        f[f"{p}_spike"]=robust_spike_flag(raw)\n\n        branch[p]=[\n            f"{p}_value",\n            f"{p}_mask",\n            f"{p}_age_h",\n            f"{p}_spike",\n        ]\n\n    meteo=[]\n\n    # Continuous meteo\n    for m,c in mcols.items():\n        if m=="WD":\n            continue\n        name=f"M_{m}"\n        f[name]=grid[c].ffill()\n        f[name+"_mask"]=grid[c].notna().astype(float)\n        meteo += [name,name+"_mask"]\n\n    # Wind direction -> u/v\n    if "WS" in mcols and "WD" in mcols:\n        ws=grid[mcols["WS"]].ffill()\n        wd=grid[mcols["WD"]].ffill()\n        th=np.deg2rad(wd)\n        f["M_wind_u"]=-ws*np.sin(th)\n        f["M_wind_v"]=-ws*np.cos(th)\n        meteo += ["M_wind_u","M_wind_v"]\n\n    # Calendar\n    hh=f.index.hour+f.index.minute/60\n    dow=f.index.dayofweek\n    f["M_hour_sin"]=np.sin(2*np.pi*hh/24)\n    f["M_hour_cos"]=np.cos(2*np.pi*hh/24)\n    f["M_dow_sin"]=np.sin(2*np.pi*dow/7)\n    f["M_dow_cos"]=np.cos(2*np.pi*dow/7)\n    meteo += ["M_hour_sin","M_hour_cos","M_dow_sin","M_dow_cos"]\n\n    return f,branch,meteo\n\n\n# ================================================================\n# 5. TRAIN-ONLY METEOROLOGICAL CONTEXT\n# ================================================================\n\n@dataclass\nclass FeatureState:\n    fill: Dict[str,float]\n    branch_scalers: Dict[str,RobustScaler]\n    meteo_scaler: RobustScaler\n    kmeans: KMeans | None\n    isolation: IsolationForest | None\n    learned_meteo_cols: List[str]\n\ndef fit_feature_state(f,branch,meteo,tr_idx):\n    tr=f.iloc[tr_idx]\n\n    fill={}\n    for c in f.columns:\n        v=tr[c].median(skipna=True)\n        fill[c]=float(v) if np.isfinite(v) else 0.0\n\n    z=f.copy()\n    for c,v in fill.items():\n        z[c]=z[c].fillna(v)\n\n    bsc={}\n    for p,cols in branch.items():\n        # only concentration and age; masks/spike flags stay interpretable\n        sc=RobustScaler(quantile_range=(10,90))\n        scale_cols=[cols[0],cols[2]]\n        sc.fit(z.iloc[tr_idx][scale_cols])\n        bsc[p]=sc\n\n    msc=RobustScaler(quantile_range=(10,90))\n    msc.fit(z.iloc[tr_idx][meteo])\n\n    learned=[\n        c for c in meteo\n        if not c.endswith("_mask")\n        and not c.endswith("_sin")\n        and not c.endswith("_cos")\n    ]\n\n    km=None\n    iso=None\n\n    if len(learned)>=2:\n        X=z.iloc[tr_idx][learned].to_numpy()\n\n        km=KMeans(\n            n_clusters=N_WEATHER_REGIMES,\n            n_init=20,\n            random_state=SEED,\n        ).fit(X)\n\n        iso=IsolationForest(\n            n_estimators=300,\n            contamination="auto",\n            random_state=SEED,\n        ).fit(X)\n\n    return FeatureState(fill,bsc,msc,km,iso,learned)\n\ndef apply_feature_state(f,branch,meteo,state,include_context=True):\n    z=f.copy()\n\n    for c,v in state.fill.items():\n        z[c]=z[c].fillna(v)\n\n    for p,cols in branch.items():\n        scale_cols=[cols[0],cols[2]]\n        z.loc[:,scale_cols]=state.branch_scalers[p].transform(z[scale_cols])\n\n    z.loc[:,meteo]=state.meteo_scaler.transform(z[meteo])\n    mout=list(meteo)\n\n    if include_context and state.kmeans is not None:\n        # Learned components use original-filled but unscaled meteorology.\n        q=f.copy()\n        for c,v in state.fill.items():\n            if c in q:\n                q[c]=q[c].fillna(v)\n\n        X=q[state.learned_meteo_cols].to_numpy()\n\n        reg=state.kmeans.predict(X)\n        for k in range(N_WEATHER_REGIMES):\n            name=f"M_regime_{k}"\n            z[name]=(reg==k).astype(float)\n            mout.append(name)\n\n        z["M_anomaly_score"]=-state.isolation.score_samples(X)\n        z["M_anomaly_flag"]=(state.isolation.predict(X)==-1).astype(float)\n        mout += ["M_anomaly_score","M_anomaly_flag"]\n\n    return z,mout\n\n\n# ================================================================\n# 6. TARGET SCALING\n# ================================================================\n\n@dataclass\nclass TargetScale:\n    center: np.ndarray\n    scale: np.ndarray\n\ndef fit_target_scale(grid,pcols,tr_idx):\n    vals=[]\n    for p in TARGETS:\n        c=pcols[p]\n        d=grid[c].shift(-HORIZON_STEPS)-grid[c]\n        vals.append(d.iloc[tr_idx].to_numpy(float))\n\n    a=np.stack(vals,axis=1)\n\n    center=[]\n    scale=[]\n\n    for j in range(a.shape[1]):\n        x=a[:,j]\n        x=x[np.isfinite(x)]\n\n        med=np.median(x) if len(x) else 0.0\n        q10=np.quantile(x,.10) if len(x) else -1\n        q90=np.quantile(x,.90) if len(x) else 1\n        sc=q90-q10\n        if sc<1e-8: sc=1.0\n\n        center.append(med)\n        scale.append(sc)\n\n    return TargetScale(np.asarray(center),np.asarray(scale))\n\ndef scale_delta(d,sc):\n    return (d-sc.center)/sc.scale\n\ndef inverse_delta(d,sc):\n    return d*sc.scale+sc.center\n\n\n# ================================================================\n# 7. WINDOWS\n# ================================================================\n\n@dataclass\nclass Pack:\n    Xg: np.ndarray\n    Xm: np.ndarray\n    y: np.ndarray\n    mask: np.ndarray\n    last: np.ndarray\n    delta: np.ndarray\n    ts: np.ndarray\n\ndef make_pack(fs,grid,idx,branch,meteo,pcols):\n    start,end=int(idx[0]),int(idx[-1])\n\n    G,M,Y,Mask,L,D,T=[],[],[],[],[],[],[]\n\n    first_target=start+HISTORY_STEPS-1+HORIZON_STEPS\n\n    for ti in range(first_target,end+1):\n        hist_end=ti-HORIZON_STEPS\n        hist_start=hist_end-HISTORY_STEPS+1\n        if hist_start<start: continue\n\n        gb=[]\n        yy=[]\n        mm=[]\n        ll=[]\n        dd=[]\n\n        for p in TARGETS:\n            gb.append(\n                fs.iloc[hist_start:hist_end+1][branch[p]].to_numpy(np.float32)\n            )\n\n            target=grid.iloc[ti][pcols[p]]\n            hist=grid.iloc[hist_start:hist_end+1][pcols[p]].dropna()\n            last=float(hist.iloc[-1]) if len(hist) else np.nan\n\n            yy.append(float(target) if pd.notna(target) else np.nan)\n            mm.append(float(pd.notna(target) and np.isfinite(last)))\n            ll.append(last)\n            dd.append(\n                float(target-last)\n                if pd.notna(target) and np.isfinite(last)\n                else np.nan\n            )\n\n        G.append(np.stack(gb,axis=1))\n        M.append(fs.iloc[hist_start:hist_end+1][meteo].to_numpy(np.float32))\n        Y.append(yy)\n        Mask.append(mm)\n        L.append(ll)\n        D.append(dd)\n        T.append(grid.index[ti])\n\n    return Pack(\n        np.asarray(G,np.float32),\n        np.asarray(M,np.float32),\n        np.asarray(Y,np.float32),\n        np.asarray(Mask,np.float32),\n        np.asarray(L,np.float32),\n        np.asarray(D,np.float32),\n        np.asarray(T),\n    )\n\n\n# ================================================================\n# 8. TABULAR HISTGB RESIDUAL BASELINE\n# ================================================================\n\ndef tab_features(pack):\n    return np.concatenate(\n        [pack.Xg.reshape(len(pack.y),-1),pack.Xm.reshape(len(pack.y),-1)],\n        axis=1\n    )\n\ndef choose_lambda(y,base,delta):\n    ok=np.isfinite(y)&np.isfinite(base)&np.isfinite(delta)\n    if ok.sum()<20:return 0.0,np.nan\n\n    best_lam=0.0\n    best=np.inf\n\n    for lam in LAMBDA_GRID:\n        pred=np.clip(base[ok]+lam*delta[ok],0,None)\n        m=mean_absolute_error(y[ok],pred)\n        if m<best:\n            best=m\n            best_lam=float(lam)\n\n    return best_lam,best\n\ndef histgb_residual(tr,va,te):\n    Xtr=tab_features(tr)\n    Xva=tab_features(va)\n    Xte=tab_features(te)\n\n    val_delta=np.full_like(va.y,np.nan,dtype=float)\n    test_delta=np.full_like(te.y,np.nan,dtype=float)\n\n    for j,p in enumerate(TARGETS):\n        ok=tr.mask[:,j].astype(bool)&np.isfinite(tr.delta[:,j])\n\n        model=HistGradientBoostingRegressor(\n            loss="absolute_error",\n            learning_rate=.05,\n            max_iter=120,\n            max_leaf_nodes=15,\n            min_samples_leaf=20,\n            l2_regularization=1.0,\n            random_state=SEED,\n        )\n        model.fit(Xtr[ok],tr.delta[ok,j])\n        val_delta[:,j]=model.predict(Xva)\n        test_delta[:,j]=model.predict(Xte)\n\n    lambdas=[]\n    pred=np.full_like(te.y,np.nan,dtype=float)\n\n    for j,p in enumerate(TARGETS):\n        lam,_=choose_lambda(\n            va.y[:,j],va.last[:,j],val_delta[:,j]\n        )\n        lambdas.append(lam)\n        pred[:,j]=np.clip(\n            te.last[:,j]+lam*test_delta[:,j],\n            0,None\n        )\n\n    return pred,lambdas\n\n\n# ================================================================\n# 9. PYTORCH DATA\n# ================================================================\n\nclass DS(Dataset):\n    def __init__(self,pack,d_scaled):\n        self.g=torch.tensor(pack.Xg,dtype=torch.float32)\n        self.m=torch.tensor(pack.Xm,dtype=torch.float32)\n        self.d=torch.tensor(np.nan_to_num(d_scaled,nan=0),dtype=torch.float32)\n        self.mask=torch.tensor(pack.mask,dtype=torch.float32)\n\n    def __len__(self):return len(self.g)\n\n    def __getitem__(self,i):\n        return self.g[i],self.m[i],self.d[i],self.mask[i]\n\n\n# ================================================================\n# 10. COMPACT CPA MODEL\n# ================================================================\n\nclass CompactPollutantEncoder(nn.Module):\n    """\n    T=4 compatible local encoder.\n    No oversized dilation is used.\n    """\n    def __init__(self,in_f,d_model=48):\n        super().__init__()\n        self.local=nn.Sequential(\n            nn.Conv1d(in_f,32,kernel_size=3,padding=1),\n            nn.GELU(),\n            nn.Conv1d(32,d_model,kernel_size=3,padding=1),\n            nn.GELU(),\n        )\n        self.norm=nn.LayerNorm(d_model)\n\n    def forward(self,x):\n        # B,T,F -> B,T,D\n        z=self.local(x.transpose(1,2)).transpose(1,2)\n        return self.norm(z)\n\n\nclass CPA(nn.Module):\n    def __init__(self,d_model=48,heads=4,enabled=True):\n        super().__init__()\n        self.enabled=enabled\n\n        if enabled:\n            self.attn=nn.MultiheadAttention(\n                d_model,heads,batch_first=True,dropout=.1\n            )\n            self.ln1=nn.LayerNorm(d_model)\n            self.ff=nn.Sequential(\n                nn.Linear(d_model,96),\n                nn.GELU(),\n                nn.Dropout(.1),\n                nn.Linear(96,d_model),\n            )\n            self.ln2=nn.LayerNorm(d_model)\n\n    def forward(self,h):\n        # B,T,P,D\n        if not self.enabled:return h\n\n        B,T,P,D=h.shape\n        q=h.reshape(B*T,P,D)\n        a,_=self.attn(q,q,q,need_weights=False)\n        q=self.ln1(q+a)\n        q=self.ln2(q+self.ff(q))\n        return q.reshape(B,T,P,D)\n\n\nclass TemporalAttention(nn.Module):\n    def __init__(self,d):\n        super().__init__()\n        self.score=nn.Sequential(\n            nn.Linear(d,64),\n            nn.Tanh(),\n            nn.Linear(64,1)\n        )\n\n    def forward(self,x):\n        a=torch.softmax(self.score(x).squeeze(-1),dim=1)\n        return (x*a.unsqueeze(-1)).sum(1),a\n\n\nclass CompactCPABiLSTM(nn.Module):\n    def __init__(\n        self,\n        ground_f,\n        meteo_f,\n        use_cpa=True,\n        bidirectional=True,\n        temporal_attention=True,\n    ):\n        super().__init__()\n\n        P=len(TARGETS)\n        D=48\n\n        self.temporal_attention=temporal_attention\n\n        self.enc=nn.ModuleList(\n            [CompactPollutantEncoder(ground_f,D) for _ in range(P)]\n        )\n\n        self.cpa=CPA(D,heads=4,enabled=use_cpa)\n\n        self.ground_proj=nn.Sequential(\n            nn.Linear(P*D,96),\n            nn.GELU()\n        )\n\n        self.meteo_proj=nn.Sequential(\n            nn.Linear(meteo_f,48),\n            nn.GELU(),\n            nn.Linear(48,48),\n            nn.GELU()\n        )\n\n        self.fusion=nn.Sequential(\n            nn.Linear(96+48,96),\n            nn.GELU(),\n            nn.LayerNorm(96)\n        )\n\n        self.rnn=nn.LSTM(\n            96,\n            96,\n            batch_first=True,\n            bidirectional=bidirectional,\n        )\n\n        out_d=192 if bidirectional else 96\n        self.pool=TemporalAttention(out_d)\n\n        self.heads=nn.ModuleList([\n            nn.Sequential(\n                nn.Linear(out_d,64),\n                nn.GELU(),\n                nn.Linear(64,1)\n            )\n            for _ in TARGETS\n        ])\n\n    def forward(self,g,m):\n        hs=[]\n        for j,e in enumerate(self.enc):\n            hs.append(e(g[:,:,j,:]))\n\n        h=torch.stack(hs,dim=2)\n        h=self.cpa(h)\n\n        B,T,P,D=h.shape\n        gg=self.ground_proj(h.reshape(B,T,P*D))\n        mm=self.meteo_proj(m)\n        z=self.fusion(torch.cat([gg,mm],dim=-1))\n\n        o,_=self.rnn(z)\n\n        if self.temporal_attention:\n            c,_=self.pool(o)\n        else:\n            c=o[:,-1]\n\n        return torch.cat([head(c) for head in self.heads],dim=1)\n\n\nclass ResidualLSTM(nn.Module):\n    def __init__(self,ground_f,meteo_f):\n        super().__init__()\n        P=len(TARGETS)\n        self.rnn=nn.LSTM(\n            P*ground_f+meteo_f,\n            96,\n            batch_first=True,\n        )\n        self.head=nn.Sequential(\n            nn.Linear(96,64),\n            nn.GELU(),\n            nn.Linear(64,P)\n        )\n\n    def forward(self,g,m):\n        B,T,P,F=g.shape\n        x=torch.cat([g.reshape(B,T,P*F),m],dim=-1)\n        o,_=self.rnn(x)\n        return self.head(o[:,-1])\n\n\n# ================================================================\n# 11. TRAINING\n# ================================================================\n\ndef masked_huber(pred,target,mask,delta=1.0):\n    e=pred-target\n    ae=e.abs()\n    l=torch.where(\n        ae<=delta,\n        .5*e**2,\n        delta*(ae-.5*delta)\n    )\n    return (l*mask).sum()/mask.sum().clamp_min(1)\n\n@torch.no_grad()\ndef eval_loss(model,loader):\n    model.eval()\n    vals=[]\n\n    for g,m,d,mask in loader:\n        g,m,d,mask=(\n            g.to(DEVICE),\n            m.to(DEVICE),\n            d.to(DEVICE),\n            mask.to(DEVICE),\n        )\n        vals.append(masked_huber(model(g,m),d,mask).item())\n\n    return float(np.mean(vals)) if vals else np.inf\n\ndef train_model(model,trds,vads,seed):\n    set_seed(seed)\n    model=model.to(DEVICE)\n\n    tr=DataLoader(trds,batch_size=BATCH_SIZE,shuffle=True)\n    va=DataLoader(vads,batch_size=BATCH_SIZE,shuffle=False)\n\n    opt=torch.optim.AdamW(\n        model.parameters(),\n        lr=LR,\n        weight_decay=WEIGHT_DECAY\n    )\n\n    best=np.inf\n    state=None\n    bad=0\n\n    for ep in range(MAX_EPOCHS):\n        model.train()\n\n        for g,m,d,mask in tr:\n            g,m,d,mask=(\n                g.to(DEVICE),\n                m.to(DEVICE),\n                d.to(DEVICE),\n                mask.to(DEVICE),\n            )\n\n            opt.zero_grad()\n            loss=masked_huber(model(g,m),d,mask)\n            loss.backward()\n            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)\n            opt.step()\n\n        vl=eval_loss(model,va)\n\n        if ep==0 or (ep+1)%10==0:\n            print(\n                f"      epoch {ep+1:02d}/{MAX_EPOCHS}: "\n                f"val_loss={vl:.6f}"\n            )\n\n        if vl<best-1e-5:\n            best=vl\n            state=deepcopy(model.state_dict())\n            bad=0\n        else:\n            bad+=1\n\n        if bad>=PATIENCE:\n            break\n\n    if state is not None:\n        model.load_state_dict(state)\n\n    return model\n\n\n@torch.no_grad()\ndef predict_delta(model,pack,scale):\n    model.eval()\n    dummy=np.zeros_like(pack.delta)\n    ds=DS(pack,dummy)\n    dl=DataLoader(ds,batch_size=256,shuffle=False)\n\n    out=[]\n\n    for g,m,d,mask in dl:\n        q=model(g.to(DEVICE),m.to(DEVICE))\n        out.append(q.cpu().numpy())\n\n    scaled=np.vstack(out)\n    return inverse_delta(scaled,scale)\n\n\n# ================================================================\n# 12. ENSEMBLE + VALIDATION GATING\n# ================================================================\n\ndef train_ensemble(\n    model_factory,\n    tr,\n    va,\n    te,\n    scale,\n    label\n):\n    dtr=scale_delta(tr.delta,scale)\n    dva=scale_delta(va.delta,scale)\n\n    trds=DS(tr,dtr)\n    vads=DS(va,dva)\n\n    val_preds=[]\n    test_preds=[]\n    models=[]\n\n    for s in range(N_SEEDS):\n        seed=SEED+s\n        print(f"   {label}: seed {seed}")\n\n        model=train_model(\n            model_factory(),\n            trds,\n            vads,\n            seed\n        )\n\n        val_preds.append(predict_delta(model,va,scale))\n        test_preds.append(predict_delta(model,te,scale))\n        models.append(model)\n\n    val_delta=np.mean(np.stack(val_preds),axis=0)\n    test_delta=np.mean(np.stack(test_preds),axis=0)\n\n    lambdas=[]\n    pred=np.zeros_like(te.y,dtype=float)\n\n    for j,p in enumerate(TARGETS):\n        lam,_=choose_lambda(\n            va.y[:,j],\n            va.last[:,j],\n            val_delta[:,j]\n        )\n        lambdas.append(lam)\n\n        pred[:,j]=np.clip(\n            te.last[:,j]+lam*test_delta[:,j],\n            0,None\n        )\n\n    return pred,lambdas,models[-1]\n\n\n# ================================================================\n# 13. METRICS\n# ================================================================\n\ndef regression_metrics(y,pred,mask,model_name):\n    rows=[]\n\n    for j,p in enumerate(TARGETS):\n        ok=(\n            mask[:,j].astype(bool)\n            &np.isfinite(y[:,j])\n            &np.isfinite(pred[:,j])\n        )\n\n        yy=y[ok,j]\n        pp=pred[ok,j]\n\n        if len(yy)<3:continue\n\n        mae=mean_absolute_error(yy,pp)\n        rmse=np.sqrt(mean_squared_error(yy,pp))\n        r2=r2_score(yy,pp)\n        denom=np.mean(np.abs(yy))\n\n        rows.append({\n            "model":model_name,\n            "pollutant":p,\n            "n":len(yy),\n            "MAE":mae,\n            "RMSE":rmse,\n            "R2":r2,\n            "nMAE":mae/denom if denom>0 else np.nan\n        })\n\n    return pd.DataFrame(rows)\n\n\ndef moving_block_boot(y,base,model,mask):\n    ok=(\n        mask.astype(bool)\n        &np.isfinite(y)\n        &np.isfinite(base)\n        &np.isfinite(model)\n    )\n\n    yy=y[ok]\n    bb=base[ok]\n    mm=model[ok]\n\n    if len(yy)<30:\n        return np.nan,np.nan,np.nan\n\n    d=np.abs(yy-bb)-np.abs(yy-mm)\n    n=len(d)\n\n    block_len=max(2,HORIZON_STEPS)\n    starts=np.arange(0,n-block_len+1)\n\n    rng=np.random.default_rng(SEED)\n    vals=np.empty(N_BOOTSTRAP)\n\n    for b in range(N_BOOTSTRAP):\n        sample=[]\n        while len(sample)<n:\n            st=int(rng.choice(starts))\n            sample.extend(d[st:st+block_len].tolist())\n        vals[b]=np.mean(sample[:n])\n\n    point=float(np.mean(d))\n    lo,hi=np.quantile(vals,[.025,.975])\n\n    return point,float(lo),float(hi)\n\n\ndef dm_abs(y,base,model,mask):\n    ok=(\n        mask.astype(bool)\n        &np.isfinite(y)\n        &np.isfinite(base)\n        &np.isfinite(model)\n    )\n\n    yy=y[ok]\n    bb=base[ok]\n    mm=model[ok]\n\n    if len(yy)<30:return np.nan,np.nan\n\n    d=np.abs(yy-bb)-np.abs(yy-mm)\n    n=len(d)\n    db=d.mean()\n    c=d-db\n\n    gamma0=np.dot(c,c)/n\n    lag=max(0,HORIZON_STEPS-1)\n\n    lrv=gamma0\n\n    for k in range(1,lag+1):\n        cov=np.dot(c[k:],c[:-k])/n\n        weight=1-k/(lag+1)\n        lrv+=2*weight*cov\n\n    if lrv<=0:return np.nan,np.nan\n\n    st=db/np.sqrt(lrv/n)\n    pv=2*(1-stats.norm.cdf(abs(st)))\n\n    return float(st),float(pv)\n\n\n# ================================================================\n# 14. EVENT METRICS\n# ================================================================\n\ndef train_event_thresholds(tr):\n    thresholds={}\n\n    for j,p in enumerate(TARGETS):\n        vals=tr.y[:,j]\n        ok=tr.mask[:,j].astype(bool)&np.isfinite(vals)\n\n        thresholds[p]=(\n            float(np.quantile(vals[ok],EVENT_QUANTILE))\n            if ok.sum()>=20 else np.nan\n        )\n\n    return thresholds\n\n\ndef event_metrics(y,pred,mask,thresholds,model_name):\n    rows=[]\n\n    for j,p in enumerate(TARGETS):\n        ok=(\n            mask[:,j].astype(bool)\n            &np.isfinite(y[:,j])\n            &np.isfinite(pred[:,j])\n        )\n\n        yy=y[ok,j]\n        pp=pred[ok,j]\n        th=thresholds[p]\n\n        if len(yy)<10 or not np.isfinite(th):\n            continue\n\n        obs=(yy>=th).astype(int)\n        fc=(pp>=th).astype(int)\n\n        tp=int(((obs==1)&(fc==1)).sum())\n        fp=int(((obs==0)&(fc==1)).sum())\n        fn=int(((obs==1)&(fc==0)).sum())\n\n        precision=precision_score(obs,fc,zero_division=0)\n        recall=recall_score(obs,fc,zero_division=0)\n        f1=f1_score(obs,fc,zero_division=0)\n        csi=tp/(tp+fp+fn) if tp+fp+fn else np.nan\n        far=fp/(tp+fp) if tp+fp else np.nan\n\n        rows.append({\n            "model":model_name,\n            "pollutant":p,\n            "threshold_train_q90":th,\n            "Precision":precision,\n            "Recall_POD":recall,\n            "F1":f1,\n            "CSI":csi,\n            "FAR":far,\n            "n_events":int(obs.sum())\n        })\n\n    return pd.DataFrame(rows)\n\n\n# ================================================================\n# 15. OPERATIONAL METRICS\n# ================================================================\n\ndef count_params(model):\n    return sum(p.numel() for p in model.parameters() if p.requires_grad)\n\ndef state_size_mb(model):\n    with tempfile.NamedTemporaryFile(suffix=".pt",delete=False) as f:\n        path=f.name\n\n    try:\n        torch.save(model.state_dict(),path)\n        return os.path.getsize(path)/(1024**2)\n    finally:\n        if os.path.exists(path):\n            os.remove(path)\n\n@torch.no_grad()\ndef latency(model,te,runs=300):\n    model.eval()\n\n    g=torch.tensor(te.Xg[:1],dtype=torch.float32,device=DEVICE)\n    m=torch.tensor(te.Xm[:1],dtype=torch.float32,device=DEVICE)\n\n    for _ in range(20):\n        _=model(g,m)\n\n    times=[]\n\n    for _ in range(runs):\n        t0=time.perf_counter()\n        _=model(g,m)\n\n        if torch.cuda.is_available():\n            torch.cuda.synchronize()\n\n        times.append((time.perf_counter()-t0)*1000)\n\n    return (\n        float(np.percentile(times,50)),\n        float(np.percentile(times,95))\n    )\n\n\n# ================================================================\n# 16. PLOTS\n# ================================================================\n\ndef plot_predictions(te,preds):\n    for j,p in enumerate(TARGETS):\n        fig,ax=plt.subplots(figsize=(11,4))\n\n        ax.plot(\n            pd.to_datetime(te.ts),\n            te.y[:,j],\n            label="Observed",\n            linewidth=1.6\n        )\n\n        for name,pred in preds.items():\n            ax.plot(\n                pd.to_datetime(te.ts),\n                pred[:,j],\n                label=name,\n                alpha=.82\n            )\n\n        ax.set_title(\n            f"{p}: +1 h point forecast, 30-min resolution, 2-h history"\n        )\n        ax.set_ylabel("µg/m³")\n        ax.legend(ncol=2)\n        fig.autofmt_xdate()\n        fig.tight_layout()\n        fig.savefig(FIG_DIR/f"{p}_final_forecast.png",dpi=300)\n        plt.close(fig)\n\n\n# ================================================================\n# 17. MAIN\n# ================================================================\n\ndef main():\n    print("="*84)\n    print("FINAL COMPACT CPA-BiLSTM — FROZEN TEMPORAL FORMULATION")\n    print("="*84)\n    print("Device:",DEVICE)\n    print(\n        f"Δt={RESAMPLE_MINUTES} min | "\n        f"T={HISTORY_STEPS} steps={HISTORY_HOURS} h | "\n        f"H={HORIZON_STEPS} steps={FORECAST_HOURS} h | POINT target"\n    )\n\n    grid,pcols,mcols=load_data()\n    tr_idx,va_idx,te_idx=split_indices(len(grid))\n\n    print(\n        "TRAIN:",grid.index[tr_idx[0]],"->",grid.index[tr_idx[-1]],\n        "| n=",len(tr_idx)\n    )\n    print(\n        "VAL  :",grid.index[va_idx[0]],"->",grid.index[va_idx[-1]],\n        "| n=",len(va_idx)\n    )\n    print(\n        "TEST :",grid.index[te_idx[0]],"->",grid.index[te_idx[-1]],\n        "| n=",len(te_idx)\n    )\n\n    base_f,branch,meteo=build_base_features(grid,pcols,mcols)\n\n    state=fit_feature_state(\n        base_f,branch,meteo,tr_idx\n    )\n\n    # Full feature set\n    fs_full,meteo_full=apply_feature_state(\n        base_f,branch,meteo,state,include_context=True\n    )\n\n    # Ablation feature set: no learned weather/anomaly context\n    fs_plain,meteo_plain=apply_feature_state(\n        base_f,branch,meteo,state,include_context=False\n    )\n\n    scale=fit_target_scale(grid,pcols,tr_idx)\n\n    tr=make_pack(fs_full,grid,tr_idx,branch,meteo_full,pcols)\n    va=make_pack(fs_full,grid,va_idx,branch,meteo_full,pcols)\n    te=make_pack(fs_full,grid,te_idx,branch,meteo_full,pcols)\n\n    tr_plain=make_pack(fs_plain,grid,tr_idx,branch,meteo_plain,pcols)\n    va_plain=make_pack(fs_plain,grid,va_idx,branch,meteo_plain,pcols)\n    te_plain=make_pack(fs_plain,grid,te_idx,branch,meteo_plain,pcols)\n\n    print(\n        "Valid forecasting windows:",\n        f"TRAIN={len(tr.y)}, VAL={len(va.y)}, TEST={len(te.y)}"\n    )\n\n    preds={}\n    lambdas={}\n\n    # Persistence\n    preds["Persistence"]=te.last.copy()\n\n    # Previous-hour mean = mean of last two 30-min observations\n    prev=np.full_like(te.y,np.nan,dtype=float)\n\n    for i in range(len(te.y)):\n        for j,p in enumerate(TARGETS):\n            vals=[]\n            # encoded sequence includes scaled values, so use original grid:\n            # approximate previous-hour mean from persistence and one prior original point.\n            target_time=pd.Timestamp(te.ts[i])\n            origin_time=target_time-pd.Timedelta(hours=FORECAST_HOURS)\n            positions=np.where(grid.index==origin_time)[0]\n\n            if len(positions):\n                oi=int(positions[0])\n                lo=max(0,oi-1)\n                z=grid.iloc[lo:oi+1][pcols[p]].dropna()\n                if len(z):\n                    vals=z.to_numpy(float)\n\n            if len(vals):\n                prev[i,j]=np.mean(vals)\n\n    prev[~np.isfinite(prev)]=te.last[~np.isfinite(prev)]\n    preds["PrevHourMean"]=prev\n\n    # HistGB\n    print("\\nTraining HistGradientBoosting residual baseline...")\n    tree_pred,tree_lam=histgb_residual(tr,va,te)\n    preds["HistGBResidual"]=tree_pred\n    lambdas["HistGBResidual"]=tree_lam\n\n    P=len(TARGETS)\n    GF=tr.Xg.shape[-1]\n    MF=tr.Xm.shape[-1]\n\n    # Residual LSTM\n    print("\\nTraining Residual-LSTM ensemble...")\n    lstm_pred,lstm_lam,lstm_model=train_ensemble(\n        lambda:ResidualLSTM(GF,MF),\n        tr,va,te,scale,\n        "Residual-LSTM"\n    )\n    preds["Residual-LSTM"]=lstm_pred\n    lambdas["Residual-LSTM"]=lstm_lam\n\n    # Full CPA\n    print("\\nTraining Compact CPA-BiLSTM ensemble...")\n    cpa_pred,cpa_lam,cpa_model=train_ensemble(\n        lambda:CompactCPABiLSTM(\n            GF,MF,\n            use_cpa=True,\n            bidirectional=True,\n            temporal_attention=True,\n        ),\n        tr,va,te,scale,\n        "Compact-CPA-BiLSTM"\n    )\n    preds["Compact-CPA-BiLSTM"]=cpa_pred\n    lambdas["Compact-CPA-BiLSTM"]=cpa_lam\n\n    # Ablation: without CPA\n    print("\\nAblation: without CPA...")\n    no_cpa_pred,no_cpa_lam,no_cpa_model=train_ensemble(\n        lambda:CompactCPABiLSTM(\n            GF,MF,\n            use_cpa=False,\n            bidirectional=True,\n            temporal_attention=True,\n        ),\n        tr,va,te,scale,\n        "without-CPA"\n    )\n    preds["without-CPA"]=no_cpa_pred\n    lambdas["without-CPA"]=no_cpa_lam\n\n    # Ablation: UniLSTM\n    print("\\nAblation: UniLSTM...")\n    uni_pred,uni_lam,uni_model=train_ensemble(\n        lambda:CompactCPABiLSTM(\n            GF,MF,\n            use_cpa=True,\n            bidirectional=False,\n            temporal_attention=True,\n        ),\n        tr,va,te,scale,\n        "UniLSTM"\n    )\n    preds["UniLSTM"]=uni_pred\n    lambdas["UniLSTM"]=uni_lam\n\n    # Ablation: no temporal attention\n    print("\\nAblation: no temporal attention...")\n    noatt_pred,noatt_lam,noatt_model=train_ensemble(\n        lambda:CompactCPABiLSTM(\n            GF,MF,\n            use_cpa=True,\n            bidirectional=True,\n            temporal_attention=False,\n        ),\n        tr,va,te,scale,\n        "no-temporal-attention"\n    )\n    preds["no-temporal-attention"]=noatt_pred\n    lambdas["no-temporal-attention"]=noatt_lam\n\n    # Ablation: no weather regimes/anomaly\n    GFp=tr_plain.Xg.shape[-1]\n    MFp=tr_plain.Xm.shape[-1]\n\n    print("\\nAblation: no weather-regime/anomaly context...")\n    plain_pred,plain_lam,plain_model=train_ensemble(\n        lambda:CompactCPABiLSTM(\n            GFp,MFp,\n            use_cpa=True,\n            bidirectional=True,\n            temporal_attention=True,\n        ),\n        tr_plain,va_plain,te_plain,scale,\n        "no-weather-context"\n    )\n    preds["no-weather-context"]=plain_pred\n    lambdas["no-weather-context"]=plain_lam\n\n    # ------------------------------------------------------------\n    # Metrics\n    # ------------------------------------------------------------\n    metric_rows=[]\n\n    for name,pred in preds.items():\n        metric_rows.append(\n            regression_metrics(te.y,pred,te.mask,name)\n        )\n\n    metrics_df=pd.concat(metric_rows,ignore_index=True)\n    metrics_df.to_csv(\n        OUT_DIR/"6_2_final_one_hour_metrics.csv",\n        index=False,\n        encoding="utf-8-sig"\n    )\n\n    print("\\nONE-HOUR METRICS")\n    print(metrics_df.to_string(index=False))\n\n    # ------------------------------------------------------------\n    # Significance: CPA vs Persistence and strongest deterministic baseline\n    # ------------------------------------------------------------\n    sig=[]\n\n    for j,p in enumerate(TARGETS):\n        psub=metrics_df[metrics_df.pollutant==p]\n\n        deterministic=[\n            "Persistence","PrevHourMean","HistGBResidual"\n        ]\n\n        strong=(\n            psub[psub.model.isin(deterministic)]\n            .sort_values("MAE")\n            .iloc[0]["model"]\n        )\n\n        for baseline in list(dict.fromkeys(["Persistence",strong,"HistGBResidual"])):\n            base=preds[baseline][:,j]\n            model=preds["Compact-CPA-BiLSTM"][:,j]\n\n            ok=(\n                te.mask[:,j].astype(bool)\n                &np.isfinite(te.y[:,j])\n                &np.isfinite(base)\n                &np.isfinite(model)\n            )\n\n            mae_b=mean_absolute_error(te.y[ok,j],base[ok])\n            mae_m=mean_absolute_error(te.y[ok,j],model[ok])\n\n            point,lo,hi=moving_block_boot(\n                te.y[:,j],\n                base,\n                model,\n                te.mask[:,j]\n            )\n\n            dm,pv=dm_abs(\n                te.y[:,j],\n                base,\n                model,\n                te.mask[:,j]\n            )\n\n            sig.append({\n                "pollutant":p,\n                "comparison":f"Compact-CPA-BiLSTM vs {baseline}",\n                "MAE_baseline":mae_b,\n                "MAE_CPA":mae_m,\n                "absolute_improvement":mae_b-mae_m,\n                "relative_improvement_pct":(\n                    100*(mae_b-mae_m)/mae_b\n                    if mae_b>0 else np.nan\n                ),\n                "bootstrap_diff":point,\n                "CI95_low":lo,\n                "CI95_high":hi,\n                "DM_stat":dm,\n                "p_value":pv,\n                "significantly_better":bool(\n                    np.isfinite(lo) and lo>0\n                    and np.isfinite(pv) and pv<.05\n                ),\n                "significantly_worse":bool(\n                    np.isfinite(hi) and hi<0\n                    and np.isfinite(pv) and pv<.05\n                )\n            })\n\n    sig_df=pd.DataFrame(sig)\n\n    sig_df.to_csv(\n        OUT_DIR/"6_2_CPA_significance.csv",\n        index=False,\n        encoding="utf-8-sig"\n    )\n\n    print("\\nCPA SIGNIFICANCE")\n    print(sig_df.to_string(index=False))\n\n    # ------------------------------------------------------------\n    # Ablation summary\n    # ------------------------------------------------------------\n    ablation_models=[\n        "Compact-CPA-BiLSTM",\n        "without-CPA",\n        "UniLSTM",\n        "no-temporal-attention",\n        "no-weather-context",\n    ]\n\n    ab=[]\n\n    for name in ablation_models:\n        z=metrics_df[metrics_df.model==name]\n\n        ab.append({\n            "Variant":name,\n            "Mean_nMAE":z.nMAE.mean(),\n            "CO_MAE":z[z.pollutant=="CO"].MAE.iloc[0],\n            "SO2_MAE":z[z.pollutant=="SO2"].MAE.iloc[0],\n            "lambda_CO":lambdas[name][0],\n            "lambda_SO2":lambdas[name][1],\n        })\n\n    ab_df=pd.DataFrame(ab)\n\n    ab_df.to_csv(\n        OUT_DIR/"6_4_ablation.csv",\n        index=False,\n        encoding="utf-8-sig"\n    )\n\n    # ------------------------------------------------------------\n    # Event metrics\n    # ------------------------------------------------------------\n    th=train_event_thresholds(tr)\n\n    ev=[]\n\n    for name in [\n        "Persistence",\n        "HistGBResidual",\n        "Residual-LSTM",\n        "Compact-CPA-BiLSTM"\n    ]:\n        ev.append(\n            event_metrics(\n                te.y,preds[name],te.mask,th,name\n            )\n        )\n\n    ev_df=pd.concat(ev,ignore_index=True)\n\n    ev_df.to_csv(\n        OUT_DIR/"6_5_high_concentration_events.csv",\n        index=False,\n        encoding="utf-8-sig"\n    )\n\n    # ------------------------------------------------------------\n    # Operational\n    # ------------------------------------------------------------\n    p50,p95=latency(cpa_model,te)\n\n    op=pd.DataFrame([\n        {\n            "Metric":"Parameter count",\n            "Value":count_params(cpa_model),\n            "Conditions":"Compact CPA-BiLSTM"\n        },\n        {\n            "Metric":"Model size MB",\n            "Value":state_size_mb(cpa_model),\n            "Conditions":"PyTorch state_dict"\n        },\n        {\n            "Metric":"Inference p50 ms",\n            "Value":p50,\n            "Conditions":f"single window; {DEVICE}"\n        },\n        {\n            "Metric":"Inference p95 ms",\n            "Value":p95,\n            "Conditions":f"single window; {DEVICE}"\n        },\n    ])\n\n    op.to_csv(\n        OUT_DIR/"6_6_operational.csv",\n        index=False,\n        encoding="utf-8-sig"\n    )\n\n    # ------------------------------------------------------------\n    # Predictions + figures\n    # ------------------------------------------------------------\n    out=pd.DataFrame({\n        "timestamp":pd.to_datetime(te.ts)\n    })\n\n    for j,p in enumerate(TARGETS):\n        out[f"{p}_observed"]=te.y[:,j]\n\n        for name,pred in preds.items():\n            out[f"{p}_{name}"]=pred[:,j]\n\n    out.to_csv(\n        OUT_DIR/"final_predictions.csv",\n        index=False,\n        encoding="utf-8-sig"\n    )\n\n    plot_predictions(\n        te,\n        {\n            "Persistence":preds["Persistence"],\n            "HistGBResidual":preds["HistGBResidual"],\n            "Residual-LSTM":preds["Residual-LSTM"],\n            "Compact-CPA-BiLSTM":preds["Compact-CPA-BiLSTM"],\n        }\n    )\n\n    metadata={\n        "temporal_formulation_status":"FROZEN FROM VALIDATION SCREENING",\n        "resample_minutes":RESAMPLE_MINUTES,\n        "history_hours":HISTORY_HOURS,\n        "history_steps":HISTORY_STEPS,\n        "forecast_hours":FORECAST_HOURS,\n        "horizon_steps":HORIZON_STEPS,\n        "target_type":"POINT",\n        "targets":TARGETS,\n        "n_seeds":N_SEEDS,\n        "validation_selected_gating":lambdas,\n        "train_period":[str(grid.index[tr_idx[0]]),str(grid.index[tr_idx[-1]])],\n        "val_period":[str(grid.index[va_idx[0]]),str(grid.index[va_idx[-1]])],\n        "test_period":[str(grid.index[te_idx[0]]),str(grid.index[te_idx[-1]])],\n        "sentinel5p_included":False,\n    }\n\n    with open(\n        OUT_DIR/"experiment_metadata.json",\n        "w",\n        encoding="utf-8"\n    ) as f:\n        json.dump(metadata,f,ensure_ascii=False,indent=2)\n\n    print("\\nABLATION")\n    print(ab_df.to_string(index=False))\n\n    print("\\nHIGH-CONCENTRATION EVENTS")\n    print(ev_df.to_string(index=False))\n\n    print("\\nOPERATIONAL")\n    print(op.to_string(index=False))\n\n    print("\\nDONE")\n    print("Results:",OUT_DIR.resolve())\n\n\nif __name__=="__main__":\n    main()\n'
base = types.ModuleType('air_quality_final_compact_cpa')
base.__dict__['__name__'] = 'air_quality_final_compact_cpa'
sys.modules['air_quality_final_compact_cpa'] = base
exec(compile(_BASE_SOURCE, 'air_quality_final_compact_cpa.py', 'exec'), base.__dict__)

# -----------------------------------------------------------------------------
# Embedded module: air_quality_sg_cpa_rnet_v2
# -----------------------------------------------------------------------------
_V2_SOURCE = '# -*- coding: utf-8 -*-\n"""\nSG-CPA-RNet v2: COMPACT RESIDUAL-GATED CROSS-POLLUTANT MODEL\n============================================================\n\nPurpose\n-------\nFinal focused experiment after the full SG-CPA-RNet ablation showed that:\n(1) compact Sentinel-5P mean context was more useful than the high-dimensional\n    satellite feature set;\n(2) unrestricted cross-pollutant mixing was unstable across pollutants/folds.\n\nThis version tests ONE focused hypothesis:\n    Cross-pollutant information should act as a bounded residual correction\n    to each pollutant\'s local representation rather than replace it.\n\nCross-pollutant update\n----------------------\nLet h_local be the pollutant-specific local representation and h_CPA the\ncross-pollutant attention representation.\n\n    correction = h_CPA - h_local\n\nProposed selective residual gate:\n    alpha_{p,t} = alpha_max * sigmoid(G_p([h_local || correction]))\n    h*_{p,t}    = h_local + alpha_{p,t} * correction\n\nThus local information is always the anchor and CPA is limited to a maximum\nfraction alpha_max of the learned cross-pollutant correction.\n\nPredefined neural variants\n--------------------------\nA0  Local-S5P-mean\n    No CPA correction (alpha = 0).\n\nA1  Residual-CPA-fixed\n    Same bounded residual CPA formulation, but alpha = alpha_max at every\n    pollutant/time step. This is the direct "without selective gate" ablation.\n\nA2  SG-CPA-RNet-v2\n    Proposed pollutant-specific adaptive residual gate:\n        0 <= alpha_{p,t} <= alpha_max.\n\nAll three neural variants use EXACTLY the same:\n- pollutant-specific compact temporal encoders;\n- compact Sentinel-5P context: CO/NO2/SO2 mean + age + availability only;\n- meteorological encoder;\n- pollutant-specific temporal decoder;\n- adaptive persistence-residual output head.\n\nBaselines\n---------\n- Persistence\n- HistGBResidual\n\nData\n----\nGround:\n    2019-10-efir_1-3.xlsx\n\nCausally aligned Sentinel-5P:\n    results_s5p_exact/s5p_aligned_30min.csv\n\nDefault evaluation\n------------------\n- 3 expanding rolling-origin folds\n- 5 neural seeds\n- alpha_max = 0.30, fixed a priori (not selected on test folds)\n\nOutputs\n-------\nresults_sg_cpa_v2/\n    v2_rolling_fold_metrics.csv\n    v2_rolling_summary.csv\n    v2_gate_summary.csv\n    v2_fold_comparisons.csv\n    v2_model_complexity.csv\n    fold*_*.csv\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nfrom copy import deepcopy\nfrom dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Dict, List, Optional, Sequence, Tuple\n\nimport numpy as np\nimport pandas as pd\n\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader, Dataset\n\nfrom sklearn.preprocessing import RobustScaler\n\nimport air_quality_final_compact_cpa as base\n\n\n# =============================================================================\n# Configuration\n# =============================================================================\nDEFAULT_SAT_CSV = Path("results_s5p_exact/s5p_aligned_30min.csv")\nDEFAULT_OUT = Path("results_sg_cpa_v2")\n\nSAT_PRODUCTS = ("CO", "NO2", "SO2")\nSAT_MAX_AGE_H = 48.0\n\n# Compact architecture: deliberately smaller than the previous full model.\nD_MODEL = 40\nMETEO_D = 32\nSAT_D = 16\nFUSION_D = 64\nDECODER_D = 48\n\n\n# =============================================================================\n# Rolling-origin folds\n# =============================================================================\ndef rolling_folds(\n    n: int, n_folds: int = 3\n) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:\n    """Expanding training interval and non-overlapping forward test intervals."""\n    if n_folds < 1:\n        raise ValueError("n_folds must be >= 1")\n\n    if n_folds == 1:\n        i1 = int(n * 0.70)\n        i2 = int(n * 0.85)\n        return [(np.arange(i1), np.arange(i1, i2), np.arange(i2, n))]\n\n    initial = 0.55\n    val_frac = 0.10\n    remaining = 1.0 - initial - val_frac\n    test_frac = remaining / n_folds\n\n    folds = []\n    for k in range(n_folds):\n        train_end = initial + k * test_frac\n        val_end = train_end + val_frac\n        test_end = 1.0 if k == n_folds - 1 else min(\n            1.0, val_end + test_frac\n        )\n\n        i1 = max(\n            int(n * train_end),\n            base.HISTORY_STEPS + base.HORIZON_STEPS + 10,\n        )\n        i2 = max(int(n * val_end), i1 + 20)\n        i3 = max(int(n * test_end), i2 + 20)\n        i3 = min(i3, n)\n\n        if i3 > i2:\n            folds.append(\n                (np.arange(i1), np.arange(i1, i2), np.arange(i2, i3))\n            )\n\n    return folds\n\n\n# =============================================================================\n# Sentinel-5P features: ONLY mean + age + availability\n# =============================================================================\ndef load_satellite_frame(\n    path: Path, grid_index: pd.DatetimeIndex\n) -> pd.DataFrame:\n    if not path.exists():\n        raise FileNotFoundError(\n            f"Satellite feature file not found: {path}\\n"\n            "Run prepare_ground_s5p_exact_time_merge_CLIPBOARD.py first."\n        )\n\n    sat = pd.read_csv(path, index_col=0, parse_dates=True)\n    sat.index = pd.to_datetime(sat.index)\n    sat = sat.sort_index().reindex(grid_index)\n    return sat\n\n\ndef compact_sat_columns(sat: pd.DataFrame) -> List[str]:\n    requested = []\n    for pol in SAT_PRODUCTS:\n        requested.extend(\n            [\n                f"S5P_{pol}_mean",\n                f"S5P_{pol}_age_h",\n                f"S5P_{pol}_available",\n            ]\n        )\n\n    missing = [c for c in requested if c not in sat.columns]\n    if missing:\n        raise RuntimeError(\n            "The exact-time S5P file is missing required compact features:\\n"\n            + "\\n".join(missing)\n        )\n    return requested\n\n\n@dataclass\nclass SatState:\n    fill: Dict[str, float]\n    scaler: RobustScaler\n    scale_cols: List[str]\n\n\ndef fit_sat_state(\n    sat: pd.DataFrame, cols: Sequence[str], tr_idx: np.ndarray\n) -> SatState:\n    """\n    All learned filling/scaling parameters are fitted on TRAIN only.\n\n    - availability -> 0\n    - age -> 60 h when unavailable/missing\n    - S5P column mean -> training median\n    - availability flags are not scaled\n    """\n    z = sat[list(cols)].copy()\n    fill: Dict[str, float] = {}\n\n    for c in cols:\n        if c.endswith("_available"):\n            fill[c] = 0.0\n        elif c.endswith("_age_h"):\n            fill[c] = SAT_MAX_AGE_H + 12.0\n        else:\n            med = z.iloc[tr_idx][c].median(skipna=True)\n            fill[c] = (\n                float(med)\n                if pd.notna(med) and np.isfinite(med)\n                else 0.0\n            )\n        z[c] = z[c].fillna(fill[c])\n\n    scale_cols = [\n        c for c in cols if not c.endswith("_available")\n    ]\n    scaler = RobustScaler(quantile_range=(10, 90))\n    scaler.fit(z.iloc[tr_idx][scale_cols])\n\n    return SatState(\n        fill=fill,\n        scaler=scaler,\n        scale_cols=scale_cols,\n    )\n\n\ndef apply_sat_state(\n    sat: pd.DataFrame, cols: Sequence[str], state: SatState\n) -> pd.DataFrame:\n    z = sat[list(cols)].copy()\n\n    for c, value in state.fill.items():\n        z[c] = z[c].fillna(value)\n\n    z.loc[:, state.scale_cols] = state.scaler.transform(\n        z[state.scale_cols]\n    )\n    return z\n\n\n# =============================================================================\n# Window packs\n# =============================================================================\n@dataclass\nclass SGPack:\n    Xg: np.ndarray\n    Xm: np.ndarray\n    Xs: np.ndarray\n    y: np.ndarray\n    mask: np.ndarray\n    last: np.ndarray\n    delta: np.ndarray\n    ts: np.ndarray\n\n\ndef make_sg_pack(\n    fs: pd.DataFrame,\n    sat_scaled: pd.DataFrame,\n    grid: pd.DataFrame,\n    idx: np.ndarray,\n    branch: Dict[str, List[str]],\n    meteo: Sequence[str],\n    pcols: Dict[str, str],\n    sat_cols: Sequence[str],\n) -> SGPack:\n    start, end = int(idx[0]), int(idx[-1])\n\n    G, M, S = [], [], []\n    Y, Mask, Last, Delta, Ts = [], [], [], [], []\n\n    first_target = (\n        start + base.HISTORY_STEPS - 1 + base.HORIZON_STEPS\n    )\n\n    for ti in range(first_target, end + 1):\n        hist_end = ti - base.HORIZON_STEPS\n        hist_start = hist_end - base.HISTORY_STEPS + 1\n\n        if hist_start < start:\n            continue\n\n        ground_branches = []\n        yy, mm, last_values, dd = [], [], [], []\n\n        for p in base.TARGETS:\n            ground_branches.append(\n                fs.iloc[hist_start : hist_end + 1][\n                    branch[p]\n                ].to_numpy(np.float32)\n            )\n\n            target = grid.iloc[ti][pcols[p]]\n            hist = (\n                grid.iloc[hist_start : hist_end + 1][pcols[p]]\n                .dropna()\n            )\n            last = (\n                float(hist.iloc[-1])\n                if len(hist)\n                else np.nan\n            )\n\n            yy.append(\n                float(target)\n                if pd.notna(target)\n                else np.nan\n            )\n            mm.append(\n                float(\n                    pd.notna(target)\n                    and np.isfinite(last)\n                )\n            )\n            last_values.append(last)\n            dd.append(\n                float(target - last)\n                if pd.notna(target) and np.isfinite(last)\n                else np.nan\n            )\n\n        # B,T,P,F after batching\n        G.append(np.stack(ground_branches, axis=1))\n\n        M.append(\n            fs.iloc[hist_start : hist_end + 1][\n                list(meteo)\n            ].to_numpy(np.float32)\n        )\n\n        S.append(\n            sat_scaled.iloc[hist_start : hist_end + 1][\n                list(sat_cols)\n            ].to_numpy(np.float32)\n        )\n\n        Y.append(yy)\n        Mask.append(mm)\n        Last.append(last_values)\n        Delta.append(dd)\n        Ts.append(grid.index[ti])\n\n    return SGPack(\n        Xg=np.asarray(G, np.float32),\n        Xm=np.asarray(M, np.float32),\n        Xs=np.asarray(S, np.float32),\n        y=np.asarray(Y, np.float32),\n        mask=np.asarray(Mask, np.float32),\n        last=np.asarray(Last, np.float32),\n        delta=np.asarray(Delta, np.float32),\n        ts=np.asarray(Ts),\n    )\n\n\n@dataclass\nclass ScaleOnly:\n    scale: np.ndarray\n\n\ndef fit_scale_only(pack: SGPack) -> ScaleOnly:\n    """\n    Robust target-delta scale fitted on training windows only.\n    """\n    scales = []\n\n    for j in range(len(base.TARGETS)):\n        x = pack.delta[:, j]\n        x = x[np.isfinite(x)]\n\n        if len(x):\n            q10, q90 = np.quantile(x, [0.10, 0.90])\n            sc = float(q90 - q10)\n        else:\n            sc = 1.0\n\n        if not np.isfinite(sc) or sc <= 1e-8:\n            sc = 1.0\n\n        scales.append(sc)\n\n    return ScaleOnly(\n        np.asarray(scales, dtype=np.float32)\n    )\n\n\nclass SGDataset(Dataset):\n    def __init__(\n        self, pack: SGPack, scale: ScaleOnly\n    ):\n        self.g = torch.tensor(\n            pack.Xg, dtype=torch.float32\n        )\n        self.m = torch.tensor(\n            pack.Xm, dtype=torch.float32\n        )\n        self.s = torch.tensor(\n            pack.Xs, dtype=torch.float32\n        )\n\n        dnorm = (\n            pack.delta / scale.scale[None, :]\n        )\n        self.d = torch.tensor(\n            np.nan_to_num(dnorm, nan=0.0),\n            dtype=torch.float32,\n        )\n        self.mask = torch.tensor(\n            pack.mask, dtype=torch.float32\n        )\n\n    def __len__(self):\n        return len(self.g)\n\n    def __getitem__(self, i):\n        return (\n            self.g[i],\n            self.m[i],\n            self.s[i],\n            self.d[i],\n            self.mask[i],\n        )\n\n\n# =============================================================================\n# SG-CPA-RNet v2\n# =============================================================================\nclass SGCPARNetV2(nn.Module):\n    """\n    cross_mode:\n        "none"  : no cross-pollutant correction\n        "fixed" : residual CPA with alpha = alpha_max\n        "gated" : pollutant-specific learned alpha in [0, alpha_max]\n    """\n\n    def __init__(\n        self,\n        ground_f: int,\n        meteo_f: int,\n        sat_f: int,\n        cross_mode: str,\n        alpha_max: float = 0.30,\n    ):\n        super().__init__()\n\n        if cross_mode not in {\n            "none", "fixed", "gated"\n        }:\n            raise ValueError(\n                f"Unknown cross_mode={cross_mode}"\n            )\n\n        if not (0.0 < alpha_max <= 1.0):\n            raise ValueError(\n                "alpha_max must be in (0, 1]."\n            )\n\n        self.cross_mode = cross_mode\n        self.alpha_max = float(alpha_max)\n        P = len(base.TARGETS)\n\n        # Pollutant-specific local temporal encoders.\n        self.encoders = nn.ModuleList(\n            [\n                base.CompactPollutantEncoder(\n                    ground_f, D_MODEL\n                )\n                for _ in range(P)\n            ]\n        )\n\n        # Same CPA block for fixed/gated variants.\n        # For the local-only variant it is not used in forward().\n        self.cpa = base.CPA(\n            D_MODEL, heads=4, enabled=True\n        )\n\n        # IMPORTANT:\n        # separate gate for each predicted pollutant.\n        # This directly addresses pollutant-dependent cross-information utility.\n        if cross_mode == "gated":\n            self.cross_gates = nn.ModuleList(\n                [\n                    nn.Sequential(\n                        nn.Linear(2 * D_MODEL, 24),\n                        nn.GELU(),\n                        nn.Linear(24, 1),\n                        nn.Sigmoid(),\n                    )\n                    for _ in range(P)\n                ]\n            )\n        else:\n            self.cross_gates = None\n\n        # Shared meteorological context.\n        self.meteo_encoder = nn.Sequential(\n            nn.Linear(meteo_f, METEO_D),\n            nn.GELU(),\n            nn.Linear(METEO_D, METEO_D),\n            nn.GELU(),\n            nn.LayerNorm(METEO_D),\n        )\n\n        # Compact satellite branch:\n        # ONLY means + age + availability.\n        # No additional high-capacity satellite gate is used.\n        self.sat_encoder = nn.Sequential(\n            nn.Linear(sat_f, SAT_D),\n            nn.GELU(),\n            nn.LayerNorm(SAT_D),\n        )\n\n        context_d = (\n            D_MODEL + METEO_D + SAT_D\n        )\n\n        self.fusion = nn.ModuleList(\n            [\n                nn.Sequential(\n                    nn.Linear(context_d, FUSION_D),\n                    nn.GELU(),\n                    nn.LayerNorm(FUSION_D),\n                )\n                for _ in range(P)\n            ]\n        )\n\n        # Pollutant-specific decoders.\n        self.decoders = nn.ModuleList(\n            [\n                nn.LSTM(\n                    FUSION_D,\n                    DECODER_D,\n                    batch_first=True,\n                    bidirectional=False,\n                )\n                for _ in range(P)\n            ]\n        )\n\n        # Deliberately no extra temporal-attention block:\n        # use the final decoder state to reduce model complexity.\n        self.delta_heads = nn.ModuleList(\n            [\n                nn.Sequential(\n                    nn.Linear(DECODER_D, 32),\n                    nn.GELU(),\n                    nn.Linear(32, 1),\n                )\n                for _ in range(P)\n            ]\n        )\n\n        # Adaptive persistence-residual coefficient remains a core component.\n        self.lambda_heads = nn.ModuleList(\n            [\n                nn.Sequential(\n                    nn.Linear(DECODER_D, 24),\n                    nn.GELU(),\n                    nn.Linear(24, 1),\n                    nn.Sigmoid(),\n                )\n                for _ in range(P)\n            ]\n        )\n\n    def _cross_update(self, local):\n        """\n        local: B,T,P,D\n        returns:\n            h     : B,T,P,D\n            alpha : B,T,P,1\n        """\n        B, T, P, _ = local.shape\n\n        if self.cross_mode == "none":\n            alpha = torch.zeros(\n                (B, T, P, 1),\n                device=local.device,\n                dtype=local.dtype,\n            )\n            return local, alpha\n\n        cpa = self.cpa(local)\n\n        # CPA already contains residual/normalization internally.\n        # We use only its correction relative to the local representation.\n        correction = cpa - local\n\n        if self.cross_mode == "fixed":\n            alpha = torch.full(\n                (B, T, P, 1),\n                self.alpha_max,\n                device=local.device,\n                dtype=local.dtype,\n            )\n\n        else:  # gated\n            alpha_parts = []\n\n            for p in range(P):\n                gate_input = torch.cat(\n                    [\n                        local[:, :, p, :],\n                        correction[:, :, p, :],\n                    ],\n                    dim=-1,\n                )\n\n                a = (\n                    self.alpha_max\n                    * self.cross_gates[p](\n                        gate_input\n                    )\n                )\n                alpha_parts.append(a)\n\n            alpha = torch.stack(\n                alpha_parts, dim=2\n            )\n\n        h = local + alpha * correction\n        return h, alpha\n\n    def forward(\n        self,\n        g,\n        m,\n        s,\n        return_diag: bool = False,\n    ):\n        # Local pollutant representations: B,T,P,D\n        local = torch.stack(\n            [\n                enc(g[:, :, j, :])\n                for j, enc in enumerate(\n                    self.encoders\n                )\n            ],\n            dim=2,\n        )\n\n        h, alpha = self._cross_update(local)\n\n        met = self.meteo_encoder(m)\n        sat = self.sat_encoder(s)\n\n        outputs = []\n        lambdas = []\n\n        for p in range(\n            len(base.TARGETS)\n        ):\n            z = torch.cat(\n                [\n                    h[:, :, p, :],\n                    met,\n                    sat,\n                ],\n                dim=-1,\n            )\n            z = self.fusion[p](z)\n\n            o, _ = self.decoders[p](z)\n\n            # Final temporal state (compact alternative to extra attention).\n            c = o[:, -1, :]\n\n            delta_raw = self.delta_heads[p](c)\n            lam = self.lambda_heads[p](c)\n\n            outputs.append(\n                lam * delta_raw\n            )\n            lambdas.append(lam)\n\n        effective_delta_norm = torch.cat(\n            outputs, dim=1\n        )\n\n        if not return_diag:\n            return effective_delta_norm\n\n        diag = {\n            "cross_alpha": alpha,\n            "lambda": torch.cat(\n                lambdas, dim=1\n            ),\n        }\n        return effective_delta_norm, diag\n\n\n# =============================================================================\n# Training and prediction\n# =============================================================================\ndef masked_huber(\n    pred,\n    target,\n    mask,\n    delta=1.0,\n):\n    e = pred - target\n    ae = e.abs()\n\n    loss = torch.where(\n        ae <= delta,\n        0.5 * e ** 2,\n        delta * (ae - 0.5 * delta),\n    )\n\n    return (\n        (loss * mask).sum()\n        / mask.sum().clamp_min(1)\n    )\n\n\ndef train_model(\n    model,\n    trds,\n    vads,\n    seed: int,\n):\n    base.set_seed(seed)\n\n    model = model.to(base.DEVICE)\n\n    tr = DataLoader(\n        trds,\n        batch_size=base.BATCH_SIZE,\n        shuffle=True,\n    )\n    va = DataLoader(\n        vads,\n        batch_size=base.BATCH_SIZE,\n        shuffle=False,\n    )\n\n    opt = torch.optim.AdamW(\n        model.parameters(),\n        lr=base.LR,\n        weight_decay=base.WEIGHT_DECAY,\n    )\n\n    best = np.inf\n    best_state = None\n    bad = 0\n\n    for _ in range(base.MAX_EPOCHS):\n        model.train()\n\n        for g, m, s, d, mask in tr:\n            g = g.to(base.DEVICE)\n            m = m.to(base.DEVICE)\n            s = s.to(base.DEVICE)\n            d = d.to(base.DEVICE)\n            mask = mask.to(base.DEVICE)\n\n            opt.zero_grad()\n\n            pred = model(g, m, s)\n            loss = masked_huber(\n                pred, d, mask\n            )\n\n            loss.backward()\n            torch.nn.utils.clip_grad_norm_(\n                model.parameters(), 1.0\n            )\n            opt.step()\n\n        model.eval()\n        val_losses = []\n\n        with torch.no_grad():\n            for g, m, s, d, mask in va:\n                g = g.to(base.DEVICE)\n                m = m.to(base.DEVICE)\n                s = s.to(base.DEVICE)\n                d = d.to(base.DEVICE)\n                mask = mask.to(base.DEVICE)\n\n                val_losses.append(\n                    float(\n                        masked_huber(\n                            model(g, m, s),\n                            d,\n                            mask,\n                        ).item()\n                    )\n                )\n\n        val_loss = (\n            float(np.mean(val_losses))\n            if val_losses\n            else np.inf\n        )\n\n        if val_loss < best - 1e-5:\n            best = val_loss\n            best_state = deepcopy(\n                model.state_dict()\n            )\n            bad = 0\n        else:\n            bad += 1\n\n            if bad >= base.PATIENCE:\n                break\n\n    if best_state is not None:\n        model.load_state_dict(\n            best_state\n        )\n\n    return model\n\n\n@torch.no_grad()\ndef predict_model(\n    model: SGCPARNetV2,\n    pack: SGPack,\n    scale: ScaleOnly,\n):\n    ds = SGDataset(pack, scale)\n    dl = DataLoader(\n        ds,\n        batch_size=256,\n        shuffle=False,\n    )\n\n    effective = []\n    alphas = []\n    lambdas = []\n\n    model.eval()\n\n    for g, m, s, d, mask in dl:\n        eff, diag = model(\n            g.to(base.DEVICE),\n            m.to(base.DEVICE),\n            s.to(base.DEVICE),\n            return_diag=True,\n        )\n\n        effective.append(\n            eff.cpu().numpy()\n        )\n        alphas.append(\n            diag["cross_alpha"]\n            .cpu()\n            .numpy()\n        )\n        lambdas.append(\n            diag["lambda"]\n            .cpu()\n            .numpy()\n        )\n\n    eff = np.vstack(effective)\n\n    delta_phys = (\n        eff * scale.scale[None, :]\n    )\n\n    pred = np.clip(\n        pack.last + delta_phys,\n        0.0,\n        None,\n    )\n\n    diagnostics = {\n        "cross_alpha": np.concatenate(\n            alphas, axis=0\n        ),\n        "lambda": np.vstack(\n            lambdas\n        ),\n    }\n\n    return pred, diagnostics\n\n\ndef train_ensemble(\n    factory,\n    tr: SGPack,\n    va: SGPack,\n    te: SGPack,\n    scale: ScaleOnly,\n    seeds: int,\n):\n    trds = SGDataset(tr, scale)\n    vads = SGDataset(va, scale)\n\n    predictions = []\n    diagnostics = []\n\n    for k in range(seeds):\n        seed = base.SEED + k\n\n        model = train_model(\n            factory(),\n            trds,\n            vads,\n            seed,\n        )\n\n        pred, diag = predict_model(\n            model,\n            te,\n            scale,\n        )\n\n        predictions.append(pred)\n        diagnostics.append(diag)\n\n    pred_ensemble = np.mean(\n        np.stack(predictions),\n        axis=0,\n    )\n\n    # Average diagnostic gates across seeds as well.\n    alpha_ensemble = np.mean(\n        np.stack(\n            [\n                d["cross_alpha"]\n                for d in diagnostics\n            ]\n        ),\n        axis=0,\n    )\n    lambda_ensemble = np.mean(\n        np.stack(\n            [\n                d["lambda"]\n                for d in diagnostics\n            ]\n        ),\n        axis=0,\n    )\n\n    return pred_ensemble, {\n        "cross_alpha": alpha_ensemble,\n        "lambda": lambda_ensemble,\n    }\n\n\n# =============================================================================\n# Metrics and diagnostics\n# =============================================================================\ndef metrics_from_pack(\n    pack: SGPack,\n    pred: np.ndarray,\n    label: str,\n    fold: int,\n) -> pd.DataFrame:\n    df = base.regression_metrics(\n        pack.y,\n        pred,\n        pack.mask,\n        label,\n    )\n    df.insert(0, "fold", fold)\n    return df\n\n\ndef gate_summary(\n    diag: dict,\n    label: str,\n    fold: int,\n    alpha_max: float,\n) -> pd.DataFrame:\n    rows = []\n\n    alpha = diag["cross_alpha"]\n    lam = diag["lambda"]\n\n    for j, pol in enumerate(\n        base.TARGETS\n    ):\n        x = alpha[:, :, j, 0].reshape(-1)\n\n        rows.append(\n            {\n                "fold": fold,\n                "model": label,\n                "gate": "cross_residual_alpha",\n                "pollutant": pol,\n                "mean": float(np.mean(x)),\n                "median": float(np.median(x)),\n                "std": float(np.std(x)),\n                "minimum": float(np.min(x)),\n                "maximum": float(np.max(x)),\n                "alpha_max": alpha_max,\n            }\n        )\n\n    for j, pol in enumerate(\n        base.TARGETS\n    ):\n        x = lam[:, j]\n\n        rows.append(\n            {\n                "fold": fold,\n                "model": label,\n                "gate": "adaptive_residual_lambda",\n                "pollutant": pol,\n                "mean": float(np.mean(x)),\n                "median": float(np.median(x)),\n                "std": float(np.std(x)),\n                "minimum": float(np.min(x)),\n                "maximum": float(np.max(x)),\n                "alpha_max": np.nan,\n            }\n        )\n\n    return pd.DataFrame(rows)\n\n\ndef trainable_parameters(model: nn.Module) -> int:\n    return int(\n        sum(\n            p.numel()\n            for p in model.parameters()\n            if p.requires_grad\n        )\n    )\n\n\ndef make_fold_comparisons(\n    metrics: pd.DataFrame,\n) -> pd.DataFrame:\n    """\n    Direct fold-level comparisons for the single predefined hypothesis.\n    Positive improvement means lower MAE for the proposed SG-CPA-RNet-v2.\n    """\n    proposed = "SG-CPA-RNet-v2"\n\n    comps = []\n\n    for comparator in [\n        "Local-S5P-mean",\n        "Residual-CPA-fixed",\n        "Persistence",\n        "HistGBResidual",\n    ]:\n        a = metrics[\n            metrics["model"] == proposed\n        ][["fold", "pollutant", "MAE"]].rename(\n            columns={"MAE": "proposed_MAE"}\n        )\n\n        b = metrics[\n            metrics["model"] == comparator\n        ][["fold", "pollutant", "MAE"]].rename(\n            columns={"MAE": "comparator_MAE"}\n        )\n\n        z = a.merge(\n            b,\n            on=["fold", "pollutant"],\n            how="inner",\n        )\n\n        z["comparator"] = comparator\n        z["MAE_improvement_pct"] = (\n            (\n                z["comparator_MAE"]\n                - z["proposed_MAE"]\n            )\n            / z["comparator_MAE"]\n            * 100.0\n        )\n\n        comps.append(z)\n\n    return pd.concat(\n        comps, ignore_index=True\n    )\n\n\n# =============================================================================\n# Main experiment\n# =============================================================================\ndef run(args):\n    args.out_dir.mkdir(\n        parents=True, exist_ok=True\n    )\n\n    base.DATA_PATH = args.ground_xlsx\n\n    grid, pcols, mcols = base.load_data()\n\n    sat = load_satellite_frame(\n        args.sat_csv,\n        grid.index,\n    )\n    sat_cols = compact_sat_columns(\n        sat\n    )\n\n    print("\\nSG-CPA-RNet v2")\n    print(\n        f"Ground rows: {len(grid):,}"\n    )\n    print(\n        "Satellite features:"\n    )\n    for c in sat_cols:\n        print(f"  {c}")\n\n    print(\n        f"alpha_max = {args.alpha_max:.2f}"\n    )\n    print(\n        "Fixed residual-CPA ablation uses "\n        "alpha = alpha_max at every time step."\n    )\n\n    fold_defs = rolling_folds(\n        len(grid), args.folds\n    )\n\n    all_metrics = []\n    all_gates = []\n    complexity_rows = []\n\n    variants = [\n        (\n            "Local-S5P-mean",\n            "none",\n        ),\n        (\n            "Residual-CPA-fixed",\n            "fixed",\n        ),\n        (\n            "SG-CPA-RNet-v2",\n            "gated",\n        ),\n    ]\n\n    for fold_no, (\n        tr_idx,\n        va_idx,\n        te_idx,\n    ) in enumerate(\n        fold_defs, start=1\n    ):\n        print(\n            f"\\n=== Rolling fold "\n            f"{fold_no}/{len(fold_defs)} ==="\n        )\n\n        # -----------------------------\n        # Ground/meteo training-only state\n        # -----------------------------\n        f, branch, meteo = (\n            base.build_base_features(\n                grid, pcols, mcols\n            )\n        )\n\n        fs_state = base.fit_feature_state(\n            f,\n            branch,\n            meteo,\n            tr_idx,\n        )\n\n        fs, meteo_out = (\n            base.apply_feature_state(\n                f,\n                branch,\n                meteo,\n                fs_state,\n                include_context=True,\n            )\n        )\n\n        # -----------------------------\n        # S5P training-only state\n        # -----------------------------\n        sat_state = fit_sat_state(\n            sat,\n            sat_cols,\n            tr_idx,\n        )\n        sat_scaled = apply_sat_state(\n            sat,\n            sat_cols,\n            sat_state,\n        )\n\n        # -----------------------------\n        # Packs\n        # -----------------------------\n        tr = make_sg_pack(\n            fs,\n            sat_scaled,\n            grid,\n            tr_idx,\n            branch,\n            meteo_out,\n            pcols,\n            sat_cols,\n        )\n        va = make_sg_pack(\n            fs,\n            sat_scaled,\n            grid,\n            va_idx,\n            branch,\n            meteo_out,\n            pcols,\n            sat_cols,\n        )\n        te = make_sg_pack(\n            fs,\n            sat_scaled,\n            grid,\n            te_idx,\n            branch,\n            meteo_out,\n            pcols,\n            sat_cols,\n        )\n\n        if min(\n            len(tr.y),\n            len(va.y),\n            len(te.y),\n        ) < 10:\n            raise RuntimeError(\n                f"Too few windows in fold {fold_no}"\n            )\n\n        # -----------------------------\n        # Persistence and HistGB baseline\n        # -----------------------------\n        tr0 = base.make_pack(\n            fs,\n            grid,\n            tr_idx,\n            branch,\n            meteo_out,\n            pcols,\n        )\n        va0 = base.make_pack(\n            fs,\n            grid,\n            va_idx,\n            branch,\n            meteo_out,\n            pcols,\n        )\n        te0 = base.make_pack(\n            fs,\n            grid,\n            te_idx,\n            branch,\n            meteo_out,\n            pcols,\n        )\n\n        persistence = te0.last.copy()\n\n        all_metrics.append(\n            metrics_from_pack(\n                te,\n                persistence,\n                "Persistence",\n                fold_no,\n            )\n        )\n\n        hist_pred, _ = base.histgb_residual(\n            tr0,\n            va0,\n            te0,\n        )\n        all_metrics.append(\n            metrics_from_pack(\n                te,\n                hist_pred,\n                "HistGBResidual",\n                fold_no,\n            )\n        )\n\n        # Target residual scaling, train only.\n        scale = fit_scale_only(tr)\n\n        ground_f = tr.Xg.shape[-1]\n        meteo_f = tr.Xm.shape[-1]\n        sat_f = tr.Xs.shape[-1]\n\n        # -----------------------------\n        # Three predefined neural variants\n        # -----------------------------\n        for label, cross_mode in variants:\n            print(\n                f"  {label} | "\n                f"cross={cross_mode} | "\n                f"alpha_max={args.alpha_max:.2f}"\n            )\n\n            def factory(\n                cm=cross_mode,\n                gf=ground_f,\n                mf=meteo_f,\n                sf=sat_f,\n            ):\n                return SGCPARNetV2(\n                    ground_f=gf,\n                    meteo_f=mf,\n                    sat_f=sf,\n                    cross_mode=cm,\n                    alpha_max=args.alpha_max,\n                )\n\n            # Complexity is architecture-specific but fold-independent.\n            if fold_no == 1:\n                tmp = factory()\n                complexity_rows.append(\n                    {\n                        "model": label,\n                        "trainable_parameters": trainable_parameters(\n                            tmp\n                        ),\n                        "alpha_max": args.alpha_max,\n                        "satellite_features": len(\n                            sat_cols\n                        ),\n                    }\n                )\n\n            pred, diag = train_ensemble(\n                factory,\n                tr,\n                va,\n                te,\n                scale,\n                args.seeds,\n            )\n\n            all_metrics.append(\n                metrics_from_pack(\n                    te,\n                    pred,\n                    label,\n                    fold_no,\n                )\n            )\n\n            all_gates.append(\n                gate_summary(\n                    diag,\n                    label,\n                    fold_no,\n                    args.alpha_max,\n                )\n            )\n\n            pred_df = pd.DataFrame(\n                {\n                    "timestamp": pd.to_datetime(\n                        te.ts\n                    )\n                }\n            )\n\n            for j, pol in enumerate(\n                base.TARGETS\n            ):\n                pred_df[\n                    f"{pol}_observed"\n                ] = te.y[:, j]\n                pred_df[\n                    f"{pol}_predicted"\n                ] = pred[:, j]\n                pred_df[\n                    f"{pol}_persistence"\n                ] = te.last[:, j]\n\n            safe_label = (\n                label.replace("+", "plus")\n                .replace(" ", "_")\n            )\n\n            pred_df.to_csv(\n                args.out_dir\n                / f"fold{fold_no}_{safe_label}.csv",\n                index=False,\n            )\n\n    # =============================================================================\n    # Save metrics\n    # =============================================================================\n    metrics = pd.concat(\n        all_metrics,\n        ignore_index=True,\n    )\n\n    metrics_path = (\n        args.out_dir\n        / "v2_rolling_fold_metrics.csv"\n    )\n    metrics.to_csv(\n        metrics_path,\n        index=False,\n    )\n\n    summary = (\n        metrics.groupby(\n            ["model", "pollutant"],\n            as_index=False,\n        )\n        .agg(\n            MAE_mean=("MAE", "mean"),\n            MAE_std=("MAE", "std"),\n            RMSE_mean=("RMSE", "mean"),\n            RMSE_std=("RMSE", "std"),\n            R2_mean=("R2", "mean"),\n            nMAE_mean=("nMAE", "mean"),\n        )\n    )\n\n    # Improvement versus persistence.\n    p_mae = (\n        summary.loc[\n            summary.model == "Persistence",\n            ["pollutant", "MAE_mean"],\n        ]\n        .rename(\n            columns={\n                "MAE_mean": "Persistence_MAE"\n            }\n        )\n    )\n\n    summary = summary.merge(\n        p_mae,\n        on="pollutant",\n        how="left",\n    )\n\n    summary[\n        "MAE_improvement_vs_persistence_pct"\n    ] = (\n        (\n            summary["Persistence_MAE"]\n            - summary["MAE_mean"]\n        )\n        / summary["Persistence_MAE"]\n        * 100.0\n    )\n\n    # Improvement versus local S5P neural control.\n    local_mae = (\n        summary.loc[\n            summary.model == "Local-S5P-mean",\n            ["pollutant", "MAE_mean"],\n        ]\n        .rename(\n            columns={\n                "MAE_mean": "Local_S5P_MAE"\n            }\n        )\n    )\n\n    summary = summary.merge(\n        local_mae,\n        on="pollutant",\n        how="left",\n    )\n\n    summary[\n        "MAE_improvement_vs_local_S5P_pct"\n    ] = (\n        (\n            summary["Local_S5P_MAE"]\n            - summary["MAE_mean"]\n        )\n        / summary["Local_S5P_MAE"]\n        * 100.0\n    )\n\n    summary_path = (\n        args.out_dir\n        / "v2_rolling_summary.csv"\n    )\n    summary.to_csv(\n        summary_path,\n        index=False,\n    )\n\n    gates = (\n        pd.concat(\n            all_gates,\n            ignore_index=True,\n        )\n        if all_gates\n        else pd.DataFrame()\n    )\n\n    gates_path = (\n        args.out_dir\n        / "v2_gate_summary.csv"\n    )\n    gates.to_csv(\n        gates_path,\n        index=False,\n    )\n\n    comparisons = make_fold_comparisons(\n        metrics\n    )\n\n    comparisons_path = (\n        args.out_dir\n        / "v2_fold_comparisons.csv"\n    )\n    comparisons.to_csv(\n        comparisons_path,\n        index=False,\n    )\n\n    complexity = pd.DataFrame(\n        complexity_rows\n    )\n\n    complexity_path = (\n        args.out_dir\n        / "v2_model_complexity.csv"\n    )\n    complexity.to_csv(\n        complexity_path,\n        index=False,\n    )\n\n    # =============================================================================\n    # Console output\n    # =============================================================================\n    print(\n        "\\n=== SG-CPA-RNet v2 rolling-origin summary ==="\n    )\n    print(\n        summary.to_string(index=False)\n    )\n\n    print(\n        "\\n=== Proposed model: fold-level MAE comparisons ==="\n    )\n    proposed_comps = (\n        comparisons.groupby(\n            ["comparator", "pollutant"],\n            as_index=False,\n        )\n        .agg(\n            improvement_mean_pct=(\n                "MAE_improvement_pct",\n                "mean",\n            ),\n            improvement_std_pct=(\n                "MAE_improvement_pct",\n                "std",\n            ),\n        )\n    )\n    print(\n        proposed_comps.to_string(\n            index=False\n        )\n    )\n\n    print(\n        "\\n=== Trainable parameters ==="\n    )\n    print(\n        complexity.to_string(index=False)\n    )\n\n    print(\n        f"\\nSaved to: "\n        f"{args.out_dir.resolve()}"\n    )\n\n    return (\n        metrics,\n        summary,\n        gates,\n        comparisons,\n        complexity,\n    )\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n\n    ap.add_argument(\n        "--ground-xlsx",\n        type=Path,\n        default=Path(\n            "2019-10-efir_1-3.xlsx"\n        ),\n    )\n\n    ap.add_argument(\n        "--sat-csv",\n        type=Path,\n        default=DEFAULT_SAT_CSV,\n    )\n\n    ap.add_argument(\n        "--out-dir",\n        type=Path,\n        default=DEFAULT_OUT,\n    )\n\n    ap.add_argument(\n        "--folds",\n        type=int,\n        default=3,\n    )\n\n    ap.add_argument(\n        "--seeds",\n        type=int,\n        default=5,\n    )\n\n    ap.add_argument(\n        "--alpha-max",\n        type=float,\n        default=0.30,\n        help=(\n            "Maximum residual CPA correction. "\n            "Fixed a priori; default 0.30."\n        ),\n    )\n\n    ap.add_argument(\n        "--quick",\n        action="store_true",\n        help=(\n            "Code/data check only: "\n            "1 rolling fold and 1 neural seed."\n        ),\n    )\n\n    args = ap.parse_args()\n\n    if args.quick:\n        args.folds = 1\n        args.seeds = 1\n\n    if not args.ground_xlsx.exists():\n        raise FileNotFoundError(\n            f"Ground workbook not found: "\n            f"{args.ground_xlsx}"\n        )\n\n    if not args.sat_csv.exists():\n        raise FileNotFoundError(\n            f"Satellite CSV not found: "\n            f"{args.sat_csv}"\n        )\n\n    run(args)\n\n\nif __name__ == "__main__":\n    main()\n'
v2 = types.ModuleType('air_quality_sg_cpa_rnet_v2')
v2.__dict__['__name__'] = 'air_quality_sg_cpa_rnet_v2'
sys.modules['air_quality_sg_cpa_rnet_v2'] = v2
exec(compile(_V2_SOURCE, 'air_quality_sg_cpa_rnet_v2.py', 'exec'), v2.__dict__)

# -----------------------------------------------------------------------------
# Final confirmatory experiment
# -----------------------------------------------------------------------------
"""
Final confirmatory experiments for BR-CPA-RNet.

Primary setting:
  BR-CPA-RNet = fixed bounded residual CPA (alpha=0.30)
  + compact S5P context, max age 48 h.

Predefined ablations/sensitivity:
  Persistence
  HistGBResidual
  Local-S5P-48h
  BR-CPA-no-S5P
  BR-CPA-RNet
  Adaptive-gated-S5P-48h
  BR-CPA-S5P-24h
  BR-CPA-S5P-12h

Statistics:
  paired moving-block bootstrap on absolute-error differences,
  stratified by rolling fold.
"""


import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd



PRIMARY_MODEL = "BR-CPA-RNet"
DEFAULT_SAT = Path("results_s5p_exact/s5p_aligned_30min.csv")
DEFAULT_OUT = Path("results_br_cpa_final")


def satellite_variant(sat, age_limit_h=None, no_satellite=False):
    z = sat.copy()

    for pol in v2.SAT_PRODUCTS:
        mean_col = f"S5P_{pol}_mean"
        age_col = f"S5P_{pol}_age_h"
        av_col = f"S5P_{pol}_available"

        if no_satellite:
            z[mean_col] = np.nan
            z[age_col] = np.nan
            z[av_col] = 0.0
            continue

        av = pd.to_numeric(z[av_col], errors="coerce").fillna(0.0) > 0.5
        age = pd.to_numeric(z[age_col], errors="coerce")

        if age_limit_h is not None:
            av = av & age.notna() & (age >= 0) & (age <= float(age_limit_h))

        z.loc[~av, mean_col] = np.nan
        z.loc[~av, age_col] = np.nan
        z[av_col] = av.astype(float)

    return z


def metrics_from_pack(pack, pred, label, fold):
    df = base.regression_metrics(pack.y, pred, pack.mask, label)
    df.insert(0, "fold", fold)
    return df


def prediction_long(fold, model, pack, pred):
    rows = []
    ts = pd.to_datetime(pack.ts)

    for j, pol in enumerate(base.TARGETS):
        valid = (
            (pack.mask[:, j] > 0.5)
            & np.isfinite(pack.y[:, j])
            & np.isfinite(pred[:, j])
        )

        for t, y, p, last in zip(
            ts[valid],
            pack.y[valid, j],
            pred[valid, j],
            pack.last[valid, j],
        ):
            rows.append(
                {
                    "fold": fold,
                    "timestamp": t,
                    "model": model,
                    "pollutant": pol,
                    "observed": float(y),
                    "predicted": float(p),
                    "persistence": float(last),
                    "abs_error": float(abs(y - p)),
                }
            )

    return pd.DataFrame(rows)


def circular_block_mean(x, block_length, rng):
    x = np.asarray(x, float)
    n = len(x)
    L = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / L))

    parts = []
    for _ in range(n_blocks):
        start = int(rng.integers(0, n))
        idx = (start + np.arange(L)) % n
        parts.append(x[idx])

    return float(np.mean(np.concatenate(parts)[:n]))


def paired_block_bootstrap(
    diff_by_fold: Dict[int, np.ndarray],
    comp_mae_by_fold: Dict[int, float],
    block_length=12,
    reps=5000,
    seed=20260827,
):
    clean = {}
    weights = {}

    for fold, x in diff_by_fold.items():
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x):
            clean[fold] = x
            weights[fold] = len(x)

    if not clean:
        return {
            "mean_abs_error_difference": np.nan,
            "improvement_pct": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "bootstrap_p_two_sided": np.nan,
            "n": 0,
        }

    total_n = int(sum(weights.values()))

    point = sum(weights[f] * np.mean(clean[f]) for f in clean) / total_n
    comp_mae = sum(
        weights[f] * comp_mae_by_fold[f] for f in clean
    ) / total_n

    improvement_pct = 100.0 * point / comp_mae if comp_mae > 0 else np.nan

    rng = np.random.default_rng(seed)
    boots = np.empty(reps, float)

    for b in range(reps):
        num = 0.0
        den = 0

        for f, x in clean.items():
            m = circular_block_mean(x, block_length, rng)
            num += weights[f] * m
            den += weights[f]

        boots[b] = num / den

    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = min(
        1.0,
        2.0 * min(np.mean(boots <= 0), np.mean(boots >= 0)),
    )

    return {
        "mean_abs_error_difference": float(point),
        "improvement_pct": float(improvement_pct),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_p_two_sided": float(p),
        "n": total_n,
    }


def significance_table(pred_long, block_length, reps):
    comparators = [
        "Persistence",
        "HistGBResidual",
        "Local-S5P-48h",
        "BR-CPA-no-S5P",
        "Adaptive-gated-S5P-48h",
    ]

    rows = []

    for pol in base.TARGETS:
        p = pred_long[
            (pred_long.model == PRIMARY_MODEL)
            & (pred_long.pollutant == pol)
        ][["fold", "timestamp", "abs_error"]].rename(
            columns={"abs_error": "primary_abs_error"}
        )

        for comp in comparators:
            c = pred_long[
                (pred_long.model == comp)
                & (pred_long.pollutant == pol)
            ][["fold", "timestamp", "abs_error"]].rename(
                columns={"abs_error": "comparator_abs_error"}
            )

            z = p.merge(c, on=["fold", "timestamp"], how="inner")

            diff_by_fold = {}
            comp_mae_by_fold = {}

            for fold, g in z.groupby("fold"):
                diff_by_fold[int(fold)] = (
                    g.comparator_abs_error.to_numpy(float)
                    - g.primary_abs_error.to_numpy(float)
                )
                comp_mae_by_fold[int(fold)] = float(
                    g.comparator_abs_error.mean()
                )

            res = paired_block_bootstrap(
                diff_by_fold,
                comp_mae_by_fold,
                block_length=block_length,
                reps=reps,
                seed=20260827 + len(rows) * 101,
            )

            rows.append(
                {
                    "primary_model": PRIMARY_MODEL,
                    "comparator": comp,
                    "pollutant": pol,
                    "block_length_steps": block_length,
                    "block_length_hours": block_length * 0.5,
                    "bootstrap_reps": reps,
                    **res,
                    "primary_better_ci95": (
                        np.isfinite(res["ci95_low"])
                        and res["ci95_low"] > 0
                    ),
                }
            )

    return pd.DataFrame(rows)


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base.DATA_PATH = args.ground_xlsx
    grid, pcols, mcols = base.load_data()

    sat_raw = v2.load_satellite_frame(args.sat_csv, grid.index)
    sat_cols = v2.compact_sat_columns(sat_raw)

    folds = v2.rolling_folds(len(grid), args.folds)

    variants = [
        ("Local-S5P-48h", "none", 48.0, False),
        ("BR-CPA-no-S5P", "fixed", None, True),
        (PRIMARY_MODEL, "fixed", 48.0, False),
        ("Adaptive-gated-S5P-48h", "gated", 48.0, False),
        ("BR-CPA-S5P-24h", "fixed", 24.0, False),
        ("BR-CPA-S5P-12h", "fixed", 12.0, False),
    ]

    all_metrics = []
    all_preds = []
    all_gates = []
    complexity_rows = []

    print("\nFINAL BR-CPA-RNet EXPERIMENT")
    print(f"Ground rows: {len(grid):,}")
    print(f"Rolling folds: {len(folds)}")
    print(f"Neural seeds: {args.seeds}")
    print(f"alpha_max: {args.alpha_max:.2f}")
    print(
        f"Bootstrap: {args.bootstrap_reps} reps, "
        f"block={args.block_length} steps "
        f"({args.block_length * 0.5:.1f} h)"
    )
    print("Primary S5P age limit: 48 h")
    print("12 h and 24 h variants are sensitivity analyses only.\n")

    for fold_no, (tr_idx, va_idx, te_idx) in enumerate(folds, start=1):
        print(f"=== Rolling fold {fold_no}/{len(folds)} ===")

        f, branch, meteo = base.build_base_features(grid, pcols, mcols)
        fs_state = base.fit_feature_state(f, branch, meteo, tr_idx)
        fs, meteo_out = base.apply_feature_state(
            f, branch, meteo, fs_state, include_context=True
        )

        tr0 = base.make_pack(fs, grid, tr_idx, branch, meteo_out, pcols)
        va0 = base.make_pack(fs, grid, va_idx, branch, meteo_out, pcols)
        te0 = base.make_pack(fs, grid, te_idx, branch, meteo_out, pcols)

        # Common 48 h reference pack.
        sat_ref = satellite_variant(sat_raw, 48.0, False)
        sat_state = v2.fit_sat_state(sat_ref, sat_cols, tr_idx)
        sat_scaled = v2.apply_sat_state(sat_ref, sat_cols, sat_state)

        te_ref = v2.make_sg_pack(
            fs, sat_scaled, grid, te_idx,
            branch, meteo_out, pcols, sat_cols
        )

        persistence = te0.last.copy()
        all_metrics.append(
            metrics_from_pack(te_ref, persistence, "Persistence", fold_no)
        )
        all_preds.append(
            prediction_long(fold_no, "Persistence", te_ref, persistence)
        )

        hist_pred, _ = base.histgb_residual(tr0, va0, te0)
        all_metrics.append(
            metrics_from_pack(te_ref, hist_pred, "HistGBResidual", fold_no)
        )
        all_preds.append(
            prediction_long(fold_no, "HistGBResidual", te_ref, hist_pred)
        )

        for label, cross_mode, age_limit, no_sat in variants:
            sat_v = satellite_variant(
                sat_raw,
                age_limit_h=age_limit,
                no_satellite=no_sat,
            )

            sat_state = v2.fit_sat_state(sat_v, sat_cols, tr_idx)
            sat_scaled = v2.apply_sat_state(sat_v, sat_cols, sat_state)

            tr = v2.make_sg_pack(
                fs, sat_scaled, grid, tr_idx,
                branch, meteo_out, pcols, sat_cols
            )
            va = v2.make_sg_pack(
                fs, sat_scaled, grid, va_idx,
                branch, meteo_out, pcols, sat_cols
            )
            te = v2.make_sg_pack(
                fs, sat_scaled, grid, te_idx,
                branch, meteo_out, pcols, sat_cols
            )

            scale = v2.fit_scale_only(tr)
            gf = tr.Xg.shape[-1]
            mf = tr.Xm.shape[-1]
            sf = tr.Xs.shape[-1]

            print(
                f"  {label} | cross={cross_mode} | "
                f"S5P={'OFF' if no_sat else str(int(age_limit)) + 'h'}"
            )

            def factory(cm=cross_mode, gf=gf, mf=mf, sf=sf):
                return v2.SGCPARNetV2(
                    ground_f=gf,
                    meteo_f=mf,
                    sat_f=sf,
                    cross_mode=cm,
                    alpha_max=args.alpha_max,
                )

            if fold_no == 1:
                tmp = factory()
                complexity_rows.append(
                    {
                        "model": label,
                        "trainable_parameters": v2.trainable_parameters(tmp),
                        "alpha_max": args.alpha_max,
                        "satellite_features": len(sat_cols),
                        "satellite_age_limit_h": (
                            np.nan if no_sat else age_limit
                        ),
                    }
                )

            pred, diag = v2.train_ensemble(
                factory, tr, va, te, scale, args.seeds
            )

            all_metrics.append(
                metrics_from_pack(te, pred, label, fold_no)
            )
            all_preds.append(
                prediction_long(fold_no, label, te, pred)
            )
            all_gates.append(
                v2.gate_summary(
                    diag, label, fold_no, args.alpha_max
                )
            )

    metrics = pd.concat(all_metrics, ignore_index=True)
    pred_long = pd.concat(all_preds, ignore_index=True)
    gates = pd.concat(all_gates, ignore_index=True)
    complexity = pd.DataFrame(complexity_rows)

    summary = (
        metrics.groupby(["model", "pollutant"], as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            nMAE_mean=("nMAE", "mean"),
        )
    )

    pmae = summary[
        summary.model == PRIMARY_MODEL
    ][["pollutant", "MAE_mean"]].rename(
        columns={"MAE_mean": "primary_MAE"}
    )

    summary = summary.merge(pmae, on="pollutant", how="left")
    summary["primary_improvement_over_model_pct"] = (
        (summary.MAE_mean - summary.primary_MAE)
        / summary.MAE_mean
        * 100.0
    )

    sig = significance_table(
        pred_long,
        block_length=args.block_length,
        reps=args.bootstrap_reps,
    )

    sensitivity = summary[
        summary.model.isin(
            [PRIMARY_MODEL, "BR-CPA-S5P-24h", "BR-CPA-S5P-12h"]
        )
    ].copy()
    sensitivity["setting_role"] = np.where(
        sensitivity.model == PRIMARY_MODEL,
        "PRIMARY",
        "SENSITIVITY_ONLY",
    )

    metrics.to_csv(args.out_dir / "final_fold_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "final_summary.csv", index=False)
    pred_long.to_csv(
        args.out_dir / "final_predictions_long.csv", index=False
    )
    sig.to_csv(
        args.out_dir / "final_bootstrap_significance.csv", index=False
    )
    sensitivity.to_csv(
        args.out_dir / "final_satellite_age_sensitivity.csv", index=False
    )
    gates.to_csv(
        args.out_dir / "final_gate_summary.csv", index=False
    )
    complexity.to_csv(
        args.out_dir / "final_model_complexity.csv", index=False
    )

    print("\n=== FINAL ROLLING-ORIGIN SUMMARY ===")
    print(
        summary[
            [
                "model",
                "pollutant",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "R2_mean",
                "nMAE_mean",
                "primary_improvement_over_model_pct",
            ]
        ].to_string(index=False)
    )

    print("\n=== PAIRED MOVING-BLOCK BOOTSTRAP ===")
    print(
        sig[
            [
                "comparator",
                "pollutant",
                "mean_abs_error_difference",
                "improvement_pct",
                "ci95_low",
                "ci95_high",
                "bootstrap_p_two_sided",
                "primary_better_ci95",
            ]
        ].to_string(index=False)
    )

    print("\nInterpretation:")
    print("  difference > 0 => BR-CPA-RNet has lower MAE.")
    print("  primary_better_ci95=True => 95% CI stays above zero.")
    print("  12 h / 24 h are sensitivity analyses only.")
    print(f"\nSaved to: {args.out_dir.resolve()}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ground-xlsx",
        type=Path,
        default=Path("2019-10-efir_1-3.xlsx"),
    )
    ap.add_argument(
        "--sat-csv",
        type=Path,
        default=DEFAULT_SAT,
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alpha-max", type=float, default=0.30)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument(
        "--block-length",
        type=int,
        default=12,
        help="30-min steps; 12 = 6 h.",
    )
    ap.add_argument("--quick", action="store_true")

    args = ap.parse_args()

    if args.quick:
        args.folds = 1
        args.seeds = 1
        args.bootstrap_reps = 300

    if not args.ground_xlsx.exists():
        raise FileNotFoundError(
            f"Ground workbook not found: {args.ground_xlsx}"
        )
    if not args.sat_csv.exists():
        raise FileNotFoundError(
            f"Satellite CSV not found: {args.sat_csv}"
        )

    run(args)


if __name__ == "__main__":
    main()
