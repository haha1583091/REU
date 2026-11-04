# -*- coding: utf-8 -*-
"""NC-REU | V9.2 minimal 3-method synthetic-truth comparison.

Reuses saved V9 and V9.1 CSVs. No synthetic regeneration, network training,
threshold search or new scale-selection optimization.
Methods: Local variance, Variogram-based, archived 6-criterion REU replay.
Calibrates three truth-proxy tolerances on the OTHER TWO seeds, evaluates on
held-out seed; repeats for each of the three seeds. This is exploratory,
not independently validated physical ground truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ('abundance_tv', 'dominant_tv', 'twopoint_maxabs')
METHODS = ('Local variance', 'Variogram-based', 'Strict REU (replay)')


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v9-dir', type=Path, default=Path(r'D:\LS_REU\results\nc_reu_V9_synthetic_truth_benchmark'))
    p.add_argument('--v91-dir', type=Path, default=Path(r'D:\LS_REU\results\nc_reu_V9_1_calibration_audit'))
    p.add_argument('--out-dir', type=Path, default=Path(r'D:\LS_REU\results\nc_reu_V9_2_3methods_compact'))
    p.add_argument('--calibration-px', type=int, default=128)
    p.add_argument('--quantile', type=float, default=0.90)
    p.add_argument('--forward', type=int, default=3)
    return p.parse_args()


def data_read(args):
    curves = pd.read_csv(args.v9_dir/'independent_truth_multiscale_q90.csv')
    choices = pd.read_csv(args.v9_dir/'Supplementary_Table_SXX_method_by_realization.csv')
    strict = pd.read_csv(args.v91_dir/'legacy_vs_six_criterion_reconstruction.csv')
    for key in ('realization','scale',*METRICS):
        if key not in curves: raise ValueError(f'Missing curves column: {key}')
    for key in ('realization','method','selected_px','seed'):
        if key not in choices: raise ValueError(f'Missing method column: {key}')
    for key in ('realization','strict_reconstructed_px'):
        if key not in strict: raise ValueError(f'Missing replay column: {key}')
    curves['scale'] = pd.to_numeric(curves['scale'])
    choices['seed'] = pd.to_numeric(choices['seed']).astype(int)
    return curves, choices, strict


def scale_pass_table(df, thresholds):
    df = df.sort_values('scale').copy()
    df['proxy_pass'] = np.logical_and.reduce([df[k].to_numpy(dtype=float) <= thresholds[k] for k in METRICS])
    # first start of three consecutive OBSERVED scales. No extrapolation.
    z = df.proxy_pass.to_numpy(dtype=bool)
    return df, z


def one_case_oracle(df, thresholds, nforward):
    d, z = scale_pass_table(df, thresholds)
    for i in range(len(d)-nforward+1):
        if bool(z[i:i+nforward].all()):
            return float(d.iloc[i]['scale']), 'confirmed_proxy'
    return np.nan, 'not_confirmed_in_observed_scales'


def at_scale(df, side, thresholds):
    if not np.isfinite(side): return None, 'unresolved_method'
    hit = df[np.isclose(df['scale'].to_numpy(dtype=float), side, atol=1e-9)]
    if hit.empty: return None, 'scale_not_in_evaluated_curve'
    vals = hit.iloc[0]
    return bool(all(float(vals[k]) <= thresholds[k] for k in METRICS)), 'evaluated_exact_scale'


def summarize(df):
    out=[]
    for meth in METHODS:
        g=df[df.method.eq(meth)]
        valid=g[g.oracle_status.eq('confirmed_proxy') & g.evaluation_status.eq('evaluated_exact_scale')]
        good=valid[valid.adequate.eq(True)]
        out.append(dict(method=meth, total_cases=len(g), oracle_confirmed=int(g.oracle_status.eq('confirmed_proxy').sum()),
                        evaluated=int(len(valid)), adequate_count=int(len(good)), false_accept_count=int(valid.false_accept.eq(True).sum()),
                        adequacy_rate_pct=100*len(good)/len(valid) if len(valid) else np.nan,
                        false_accept_rate_pct=100*valid.false_accept.eq(True).sum()/len(valid) if len(valid) else np.nan,
                        median_scale_error_pct=100*valid.scale_rel_error.median() if len(valid) else np.nan,
                        median_area_overhead_pct=good.area_overhead_pct.median() if len(good) else np.nan))
    return pd.DataFrame(out)


def main():
    a=cli()
    if not 0<a.quantile<1: raise ValueError('--quantile must be in (0,1)')
    if a.forward<1: raise ValueError('--forward must be >=1')
    curves, choices, strict=data_read(a)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    # Every realization must appear in methods, truth, and replay.
    common=set(curves.realization)&set(choices.realization)&set(strict.realization)
    cases=choices[choices.realization.isin(common)][['realization','seed']].drop_duplicates()
    seeds=sorted(cases.seed.unique())
    if len(seeds)<3: raise ValueError('Expected >=3 independent seeds for heldout evaluation')
    results=[]; calibrations=[]
    for test_seed in seeds:
        train_cases=cases[cases.seed.ne(test_seed)].realization
        train=curves[curves.realization.isin(train_cases) & curves.scale.eq(a.calibration_px)]
        missing=set(train_cases)-set(train.realization)
        if missing: raise ValueError(f'Calibration scale {a.calibration_px} missing in {sorted(missing)}')
        thresholds={k:float(train[k].quantile(a.quantile)) for k in METRICS}
        calibrations.append({'test_seed':int(test_seed), 'n_training_realizations':len(train_cases),
                             'calibration_px':int(a.calibration_px),**{f'threshold_{k}':v for k,v in thresholds.items()}})
        print(f'[HELDOUT seed={test_seed}] calibration={thresholds}')
        test_cases=cases[cases.seed.eq(test_seed)].realization
        for case in test_cases:
            df=curves[curves.realization.eq(case)].copy()
            oracle,status=one_case_oracle(df,thresholds,a.forward)
            if len(df)==0: continue
            for meth in METHODS:
                if meth == 'Strict REU (replay)':
                    row=strict[strict.realization.eq(case)]
                    side=float(row.iloc[0].strict_reconstructed_px) if len(row) else np.nan
                else:
                    row=choices[choices.realization.eq(case)&choices.method.eq(meth)]
                    side=float(row.iloc[0].selected_px) if len(row) else np.nan
                passed,evalstatus=at_scale(df,side,thresholds)
                confirmed=np.isfinite(oracle)
                # Results are undefined if empirical proxy oracle is unresolved.
                adequate=bool(passed) if (confirmed and passed is not None) else np.nan
                is_false=(not passed) if (confirmed and passed is not None) else np.nan
                error=abs(side-oracle)/oracle if confirmed and np.isfinite(side) else np.nan
                overhead=100*((side/oracle)**2-1) if confirmed and passed is True and np.isfinite(side) else np.nan
                results.append(dict(realization=case,seed=int(test_seed),method=meth,selected_px=side,
                                    oracle_px=oracle,oracle_status=status,evaluation_status=evalstatus,
                                    adequate=adequate,false_accept=is_false,scale_rel_error=error,
                                    area_overhead_pct=overhead))
    detail=pd.DataFrame(results)
    # Convert nullable outcomes to numerical-friendly values, retaining NaNs.
    for k in ('adequate','false_accept'):
        detail[k]=pd.to_numeric(detail[k],errors='coerce')
    report=summarize(detail)
    detail.to_csv(a.out_dir/'Table_by_realization.csv',index=False,encoding='utf-8-sig')
    report.to_csv(a.out_dir/'Supplementary_Table_3methods.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(calibrations).to_csv(a.out_dir/'calibration_audit.csv',index=False,encoding='utf-8-sig')
    with (a.out_dir/'limitations.json').open('w',encoding='utf-8') as f:
        json.dump({'confirmatory':False,'strict_selection':'archived six-criterion REPLAY (not verified exact V2.7 core call)',
                   'calibration':'fixed 128 px reference, leave-one-seed-out; empirically defined proxy, not physical ground truth',
                   'baseline_selection':'frozen V9 PC1 local variance / Euclidean variogram',
                   'oracle':'minimum of three consecutive observed scales passing all three synthetic true-state proxies',
                   'reference_overlap':'truth global field includes windows; no new simulation run',
                   'no_result_driven_threshold_tuning':True},f,indent=2,ensure_ascii=False)
    print('\n',report.to_string(index=False))
    print('\nSAVED:',a.out_dir)

if __name__=='__main__':main()
