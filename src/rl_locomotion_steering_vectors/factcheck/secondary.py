"""Independent geometry and resampling of saved observations, with no policy imports.

Scalar-first quaternion yaw follows the ZYX rotation identity. Resampling uses
https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats.bootstrap.html
and NumPy's documented quantile, arctan2 and unwrap functions. These computations
test recorded arithmetic, not the simulator or the semantic identity of an axis.
"""
import numpy as np
from scipy import stats

from .numerical import episode_summary

SEED = 20261016


def statistics(values, seed=SEED, resamples=10000):
    """Reduce whole reset differences; exact sign-flip inference needs sign symmetry."""
    x = np.asarray(values, dtype=float)
    out = dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), sd=float(x.std(ddof=1)),
               positive_pairs=int((x > 0).sum()), negative_pairs=int((x < 0).sum()), zero_pairs=int((x == 0).sum()),
               resamples=resamples, seed=seed)
    for method, label in [('BCa', 'bca_ci'), ('percentile', 'percentile_ci')]:
        interval = stats.bootstrap((x,), np.mean, method=method, n_resamples=resamples, random_state=seed).confidence_interval
        out[label] = [float(interval.low), float(interval.high)]
    out['sign_flip_p'] = float(stats.permutation_test((x,), np.mean, permutation_type='samples', n_resamples=resamples, random_state=seed).pvalue)
    nonzero = x[x != 0]
    out['wilcoxon_p'] = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.
    return out


def crossed_interval(matrix, seed=SEED, resamples=10000):
    """Resample directions and resets independently while preserving their crossed cells."""
    x = np.asarray(matrix, dtype=float)
    rng = np.random.default_rng(seed)
    rows = rng.integers(x.shape[0], size=(resamples, x.shape[0]))
    columns = rng.integers(x.shape[1], size=(resamples, x.shape[1]))
    draws = [x[np.ix_(r, c)].mean() for r, c in zip(rows, columns)]
    return dict(mean=float(x.mean()), ci=np.quantile(draws, [.025, .975]).tolist(), resamples=resamples, seed=seed)


def joint_interval(arrays, *, absolute=False):
    """Seed bootstrap of a sum, or the difference of two absolute mean effects."""
    values = np.asarray(arrays)
    indices = np.random.default_rng(SEED).integers(values.shape[1], size=(10000, values.shape[1]))
    means = values[:, indices].mean(axis=2)
    draws = np.abs(means[0])-np.abs(means[1]) if absolute else means.sum(axis=0)
    return np.quantile(draws, [.025, .975]).tolist()


def pair_geometry(left, right, warmup=100):
    """Use common surviving transitions; duration and inclusion remain treatment-dependent."""
    n = min(int(left['length']), int(right['length']))-warmup
    result = dict(common_steps=max(0, n))
    if n < 100:
        return result
    yaw, vx, vy, angle = [], [], [], []
    for ep in (left, right):
        w, x, y, z = np.asarray(ep['observations'], dtype=float)[:, 1:5].T
        yaw.append(np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))[warmup:warmup+n])
        vx.append(ep['x_velocity'][warmup:warmup+n])
        vy.append(ep['y_velocity'][warmup:warmup+n])
        angle.append(np.arctan2(vy[-1].sum(), vx[-1].sum()))
    offset = yaw[0].mean()-angle[0]
    lateral = [np.mean(-np.sin(h-offset)*x+np.cos(h-offset)*y) for h,x,y in zip(yaw,vx,vy)]
    window_yaw = [float(np.degrees(yaw[1][s:s+100].mean()-yaw[0][s:s+100].mean())) for s in range(0, n-99, 100)]
    window_y = [float(vy[1][s:s+100].mean()-vy[0][s:s+100].mean()) for s in range(0, n-99, 100)]
    t = np.arange(len(window_yaw), dtype=float)
    centered = t-t.mean()
    slope = float(centered@np.asarray(window_yaw)/(centered@centered)) if len(t) > 1 else 0.
    displacement = [float(ep['y_velocity'][warmup:].sum()*float(ep['dt'])) for ep in (left,right)]
    return result | dict(delta_y_velocity=float(vy[1].mean()-vy[0].mean()), delta_y_displacement=displacement[1]-displacement[0],
                         delta_travel_angle_deg=float(np.degrees(angle[1]-angle[0])), delta_mean_yaw_deg=float(np.degrees(yaw[1].mean()-yaw[0].mean())),
                         delta_gait_lateral_velocity=float(lateral[1]-lateral[0]), yaw_slope_deg_per_100_steps=slope,
                         window_delta_yaw_deg=window_yaw, window_delta_y_velocity=window_y)


def geometry(left, right):
    """Apply the historical two-window inclusion rule and report its denominator explicitly."""
    pairs = [pair_geometry(a,b) for a,b in zip(left,right)]
    usable = [p for p in pairs if p['common_steps'] >= 200]
    keys = ('delta_y_velocity','delta_y_displacement','delta_travel_angle_deg','delta_mean_yaw_deg','delta_gait_lateral_velocity','yaw_slope_deg_per_100_steps')
    windows = []
    for i in range(9):
        group = [p for p in usable if len(p['window_delta_yaw_deg']) > i]
        if len(group) >= 5:
            windows.append(dict(start=100+100*i, pairs=len(group), delta_yaw_deg=statistics([p['window_delta_yaw_deg'][i] for p in group], resamples=2000),
                                delta_y_velocity=statistics([p['window_delta_y_velocity'][i] for p in group], resamples=2000)))
    return dict(pairs=len(usable), windows=windows, **{key: statistics([p[key] for p in usable]) for key in keys})


def robustness(left, right):
    """Compute alternative estimands without changing the historical primary denominator."""
    before, after = ([episode_summary(x,100) for x in arm] for arm in (left,right))
    a,b = (np.array([r['lateral'] for r in arm]) for arm in (before,after))
    d = b-a
    available = np.array([not r['prefix_only'] for r in before])
    survive = np.array([not (x['failure'] or y['failure']) for x,y in zip(before,after)])
    distance = np.array([(y['y_velocity'][100:].sum()*float(y['dt'])-x['y_velocity'][100:].sum()*float(x['dt'])) for x,y in zip(left,right)])
    loo = (d.sum()-d)/(len(d)-1)
    return dict(recorded=statistics(d), excluding_prefix_only=statistics(d[available]), both_arms_survive=statistics(d[survive]),
                y_displacement_m=statistics(distance), trimmed_mean_10pct=float(np.sort(d)[int(.1*len(d)):-int(.1*len(d))].mean()),
                leave_one_out_range=[float(loo.min()),float(loo.max())], arm_correlation=dict(all_pairs=float(np.corrcoef(a,b)[0,1]), excluding_prefix_only=float(np.corrcoef(a[available],b[available])[0,1])),
                variance_ratio_paired_vs_unpaired=float(d.var(ddof=1)/(a.var(ddof=1)+b.var(ddof=1))))


def ensemble_statistics(delta, arms, directions):
    """Recompute P2/S1/S2/M1/M2 arithmetic; calibrated p-value assumptions are separate."""
    names = sorted(directions)
    odd = {n:(delta[n+'__+0.1000']-delta[n+'__-0.1000'])/2 for n in names}
    even = {n:(delta[n+'__+0.1000']+delta[n+'__-0.1000'])/2 for n in names}
    randoms = [n for n in names if n.startswith('random')]
    shuffled = [n for n in names if n.startswith('shuffled')]
    r = np.abs([odd[n].mean() for n in randoms])
    rank = lambda value, group=r: float((1+np.count_nonzero(group >= value))/(1+len(group)))
    primary = abs(odd['lateral'].mean())
    p1 = statistics(delta['lateral__-0.1000'])
    p2 = dict(primary_abs_odd=float(primary), primary_odd_tests=statistics(odd['lateral']), random_abs_odd_quantiles=np.quantile(r,[.5,.9,.95,.99]).tolist(), rank_p=rank(primary),
              best_arm_rank_p=rank(delta['lateral__-0.1000'].mean(),np.array([max(directions[n]['minus_arm'],directions[n]['plus_arm']) for n in randoms])))
    p = np.array([p1['sign_flip_p'],p2['rank_p']]); order = np.argsort(p)
    adjusted = np.empty(2); adjusted[order] = np.minimum(1,np.maximum.accumulate(p[order]*[2,1]))
    p2.update(holm_adjusted=dict(P1=float(adjusted[0]),P2=float(adjusted[1])),falsified=bool(adjusted[1]>.05))
    s = np.abs([odd[n].mean() for n in shuffled])
    s1 = dict(turning_abs_odd=float(abs(odd['turning'].mean())), turning_minus_primary_abs_odd_ci=joint_interval([odd['turning'],odd['lateral']],absolute=True), shuffled_abs_odd=s.tolist(), shuffled_rank_p=rank(primary,s))
    s1['label_specific'] = bool(s1['turning_minus_primary_abs_odd_ci'][1]<0 and s1['shuffled_rank_p']<=.05)
    s2 = dict(random_even=crossed_interval([even[n] for n in randoms]),random_odd_sanity=crossed_interval([odd[n] for n in randoms]),
              random_minus_arm_mean=float(np.mean([directions[n]['minus_arm'] for n in randoms])),random_plus_arm_mean=float(np.mean([directions[n]['plus_arm'] for n in randoms])))
    m1 = dict(parallel_odd=float(odd['heading_parallel'].mean()),orthogonal_odd=float(odd['heading_orthogonal'].mean()),additivity_gap_ci=joint_interval([odd['heading_parallel'],odd['heading_orthogonal'],-odd['lateral']]),
              orthogonal_renorm_abs_odd=float(abs(odd['heading_orthogonal_renorm'].mean())),orthogonal_renorm_rank_p=rank(abs(odd['heading_orthogonal_renorm'].mean())),axis_renorm_odd=float(odd['heading_axis_renorm'].mean()),axis_renorm_rank_p=rank(abs(odd['heading_axis_renorm'].mean())))
    m1['falsified'] = bool(m1['orthogonal_renorm_rank_p']<=.05 or np.sign(m1['parallel_odd'])!=np.sign(odd['lateral'].mean()) or abs(m1['parallel_odd'])<.5*primary)
    lo,hi = (delta[f'perceived_heading__{sign}0.160876rad'] for sign in ('-','+'))
    m2 = dict(minus_delta_arm=statistics(lo),plus_delta_arm=statistics(hi),odd=statistics((hi-lo)/2),even_mean=float(((hi+lo)/2).mean()),odd_ratio_to_primary=float(((hi-lo)/2).mean()/odd['lateral'].mean()),failures=[arms[f'perceived_heading__{sign}0.160876rad']['failures'] for sign in ('-','+')])
    m2['verdict'] = 'supported' if m2['odd']['bca_ci'][1]<0 else 'falsified' if m2['odd']['bca_ci'][0]>0 else 'inconclusive'
    return dict(P1=p1,P2=p2,S1=s1,S2=s2,M1=m1,M2=m2)
