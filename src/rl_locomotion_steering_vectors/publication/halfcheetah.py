"""Bind the paper's HalfCheetah results to saved JSON, preserving all selected controls.

These are display reductions of recorded results, independently checked by factcheck.
No production estimator or policy is executed here.
"""
import json


def context(root):
    """Return empirical prose fields and a complete locked-confirmation table."""
    load=lambda name:json.loads((root/'reports'/name/'results.json').read_text(encoding='utf-8'))
    one,two,three,four=(load(n) for n in ('hc-classic-001','hc-running-002','hc-height-speed-003','hc-height-speed-004'))
    diagnostic=json.loads((root/'reports/hc-running-002/vector_diagnostics.json').read_text(encoding='utf-8'))
    validation=four['validation'];confirmation=four['confirmation']
    selected=next(x for x in confirmation if x['vector']=='speed')
    effect=selected['effect'];n=effect['n_pairs']
    passed=lambda rows,primary:sum(x['effect']['gate'] for x in rows if (x['vector']==x['behavior'])==primary)
    rows=['| Locked condition | Speed change | 95% interval (m/s) | Failures | Pilot gate |','| --- | ---: | --- | ---: | --- |']
    for row in confirmation:
        e=row['effect'];ci=', '.join(f'{v:.3f}' for v in e['ci95'])
        rows.append(f"| {row['vector']} ({row['alpha']:+g}) | {100*e['relative_effect']:+.2f}% | [{ci}] | {round(e['treated_means']['failure']*e['n_pairs'])}/{e['n_pairs']} | {'pass' if e['gate'] else 'fail'} |")
    changes=[(r['effect']['treated_means']['speed']/r['effect']['baseline_means']['speed']-1)*100 for run in (one,two,three,four) for phase in ('validation','confirmation') for r in run.get(phase,[])]
    return dict(hc001_status=one['status'],hc002_status=two['status'],hc003_status=three['status'],hc004_status=four['status'],
                hc002_eligible=diagnostic['windows']['eligible_windows'],hc002_total=diagnostic['windows']['total_windows'],hc002_slow=diagnostic['windows']['excluded_slow_windows'],
                hc002_validation=len(two['validation']),hc002_primary_pass=passed(two['validation'],True),hc002_control_pass=passed(two['validation'],False),
                hc003_validation=len(three['validation']),hc003_primary_pass=passed(three['validation'],True),hc003_control_pass=passed(three['validation'],False),
                hc004_validation=len(validation),hc004_primary_pass=passed(validation,True),hc004_alpha=f"{selected['alpha']:+g}",hc004_n=n,
                hc004_effect=f"{effect['effect']:+.3f}",hc004_ci='['+', '.join(f'{v:.3f}' for v in effect['ci95'])+']',
                hc004_failures=round(effect['treated_means']['failure']*n),hc004_baseline_failures=round(effect['baseline_means']['failure']*n),hc004_table='\n'.join(rows),
                hc_evaluations=len(changes),hc_max_speed_increase=f'{max(changes):+.2f}')
