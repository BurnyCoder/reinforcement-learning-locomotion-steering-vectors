"""Inspect source history, cached identities and checkpoint JSON without loading policies.

Git objects identify historical source bytes; ZIP JSON metadata is not executable.
https://git-scm.com/docs/git-show ; https://docs.python.org/3.12/library/zipfile.html .
Commit/log timestamps establish recorded chronology, not trusted wall-clock attestation.
"""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np

from .audit import EXPERIMENTS, condition, digest, read_json
from .numerical import contrast, fitting


def git(root, *args):
    """Read Git objects without modifying the checkout."""
    return subprocess.check_output(['git', *args], cwd=root).decode('utf-8').replace('\r\n','\n')


def source_versions(root, name):
    """Map historical normalized source fingerprints to the commits containing those bytes."""
    path = 'src/rl_locomotion_steering_vectors/'+name+'.py'
    revisions = git(root,'log','3fb62bc','--format=%H','--',path).splitlines()
    values = {}
    for revision in revisions:
        source = git(root,'show',revision+':'+path)
        values.setdefault(hashlib.sha256(source.encode()).hexdigest(), []).append(revision)
    return values


def provenance(root: Path):
    """Record successful checks and provenance gaps explicitly, never infer freshness from a name."""
    runtime, perception = source_versions(root,'runtime'), source_versions(root,'perception')
    cohorts, identities, logs, protocols, checkpoints = {}, {}, {}, {}, {}
    for name in EXPERIMENTS:
        run = root/'runs'/name
        manifest = read_json(run/'manifest.json')
        split = manifest['seed_splits']
        cohorts[name] = dict(reserved=split, observed={p.name:sorted({int(ep.stem) for ep in p.rglob('*.npz')}) for p in (run/'episodes').iterdir() if p.is_dir()}, model_key=manifest['model_key'])
        cohorts[name]['within_manifest_overlap'] = {a+'/'+b: sorted(set(split[a])&set(split[b])) for a in split for b in split if a<b and set(split[a])&set(split[b])}
        if (run/'retarget.json').exists():
            cohorts[name]['reused_fitting'] = 'hc-running-002:fit (deliberate reuse; not an independent extraction)'
        key = manifest['model_key']
        if key not in checkpoints:
            meta = manifest['provenance']
            checkpoint = root/'artifacts/base'/key/meta['filename']
            with zipfile.ZipFile(checkpoint) as archive:
                data = json.loads(archive.read('data'))
            checksum = digest(checkpoint)
            if checksum != meta['sha256']:
                raise ValueError('Checkpoint bytes differ from recorded hash')
            checkpoints[key] = dict(path=checkpoint.relative_to(root).as_posix(), sha256=checksum, revision=meta['revision'],
                                    metadata={k:data.get(k) for k in ('seed','num_timesteps','learning_rate','gamma','batch_size','buffer_size','policy_kwargs')},
                                    limitation='JSON metadata inspected only; parameter hashes, network execution and upstream training not revalidated')
        for path in sorted(run.glob('*.log')):
            content = path.read_text(encoding='utf-8')
            times = re.findall(r'(?m)^\d{4}-\d\d-\d\d[^\r\n]*',content)
            events = [line for line in content.splitlines() if any(word in line.lower() for word in ('traceback','error','selected','selection','confirmation','rejected'))]
            logs[path.relative_to(root).as_posix()] = dict(first=times[0][:100] if times else None,last=times[-1][:100] if times else None,
                                                         event_lines=events[:3]+events[-3:] if len(events)>6 else events,event_count=len(events),sha256=digest(path))
    for path in sorted((root/'runs').glob('*/episodes/*/*/identity.json')):
        record = read_json(path)
        history = perception if 'perception_sha256' in record else runtime
        fingerprint = record.get('perception_sha256',record.get('runtime_sha256'))
        identities[path.relative_to(root).as_posix()] = dict(record=record,source_commits=history.get(fingerprint,[]),
                                                            source_status='matched_git_source' if fingerprint in history else 'unresolved_source_snapshot')
    for path in sorted((root/'reports').glob('*preregistration.md')):
        relative = path.relative_to(root).as_posix()
        history = git(root,'log','3fb62bc','--reverse','--format=%H|%aI|%cI','--',relative).splitlines()
        protocols[relative] = dict(first_git_record=history[0] if history else None, current_sha256=digest(path),
                                   limitation='Git dates do not prove push time or absence of unrecorded prior observations')
    overlaps = {}
    for a,ca in cohorts.items():
        for b,cb in cohorts.items():
            if a<b and ca['model_key']==cb['model_key']:
                intersection = set().union(*map(set,ca['observed'].values())) & set().union(*map(set,cb['observed'].values()))
                if intersection:
                    overlaps[a+'/'+b] = sorted(intersection)
    audit_seeds = sorted(int(p.stem) for p in (root/'runs/ant-005-independent-audit/episodes/audit/baseline').glob('*.npz'))
    cfg = read_json(root/'runs/ant-classic-005/manifest.json')['config']
    application = set(range(cfg['seed_offset']+40000,cfg['seed_offset']+40010))
    return dict(cohorts=cohorts,cross_run_observed_overlap=overlaps,audit_application_overlap=sorted(application&set(audit_seeds)),
                identities=identities,protocols=protocols,logs=logs,checkpoints=checkpoints,
                environment={name:version(name) for name in ('numpy','scipy','matplotlib','reportlab','huggingface-hub')})


def direction_algebra(root: Path):
    """Reconstruct saved controls and projections; neural response axes remain observed inputs."""
    folder = root/'reports/ant-classic-005-independent-audit'
    meta = read_json(folder/'directions.json')
    with np.load(folder/'directions.npz',allow_pickle=False) as archive:
        vectors = {k:archive[k].astype(float) for k in archive.files}
    primary, yaw = vectors['lateral'], vectors['g_yaw']
    norm = np.linalg.norm(primary); unit = yaw/np.linalg.norm(yaw)
    parallel = (primary@unit)*unit; orthogonal = primary-parallel
    reference = dict(lateral=primary,heading_parallel=parallel,heading_orthogonal=orthogonal,
                     heading_orthogonal_renorm=orthogonal*norm/np.linalg.norm(orthogonal),heading_axis_renorm=unit*norm)
    draws = np.random.default_rng(np.random.SeedSequence(meta['random_seed'])).standard_normal((99,len(primary)))
    draws *= norm/np.linalg.norm(draws,axis=1,keepdims=True)
    reference.update({f'random{i:03d}':v for i,v in enumerate(draws)})
    episodes = condition(root/'runs/ant-classic-005/episodes/fit/baseline',activation=True)
    table,_,scale = fitting(episodes,warmup=100,fraction=.5)
    for seed in meta['shuffle_seeds']:
        rng = np.random.default_rng(np.random.SeedSequence([seed,3]))
        rng.permutation(np.unique(table['episode']))
        rng.standard_normal((3,len(primary)))
        raw,_ = contrast(table,'lateral',labels=rng.permutation(table['lateral']))
        reference[f'shuffled{seed:02d}'] = raw*scale/np.linalg.norm(raw)
    errors = {name:float(np.max(np.abs(vectors[name]-value.astype(np.float32)))) for name,value in reference.items()}
    if max(errors.values())>1e-7:
        raise ValueError(f'Audit direction reconstruction failed: {errors}')
    properties = {name:dict(norm=float(np.linalg.norm(value)),cosine_with_primary=float(value@primary/np.linalg.norm(value)/norm),
                            projection_g_yaw=float(value@unit),projection_g_omega=float(value@vectors['g_omega']/np.linalg.norm(vectors['g_omega']))) for name,value in vectors.items() if not name.startswith('g_')}
    return dict(reconstruction_max_abs_error=errors,properties=properties,delta_eq_rad=float(.1*(primary@unit)/np.linalg.norm(yaw)),
                limitation='The origin and finite-difference accuracy of saved g_yaw/g_omega require model execution; only their algebraic use is checked here.')
