"""Render the claim ledger and dated correction report from explicit evidence dispositions.

This inventories prose and code comments without executing their examples. A group
verdict is deliberately bounded by its named checks; uncovered material is unresolved.
"""
from collections import Counter
from fnmatch import fnmatch
import io
from pathlib import Path
import re
import tokenize

from .audit import digest, read_json
from .inventory import encode
from .provenance import git

OUT = Path('reports/fact-check-2026-10-29')


def text_claims(path):
    """Locate complete Markdown blocks or Python comments/docstrings for reviewable traceability."""
    source = path.read_text(encoding='utf-8')
    if path.suffix == '.py':
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT or token.type == tokenize.STRING and token.string.startswith(('"""',"'''")):
                yield token.start[0], token.string
    elif path.suffix in ('.md','.in'):
        start, block = 1, []
        for line, content in enumerate(source.splitlines()+[''],1):
            if not content.strip():
                if block:
                    yield start,'\n'.join(block)
                    block=[]
            else:
                if not block: start=line
                block.append(content)


def ledger(root, inventory):
    """Assign all inventoried surfaces a disposition; never upgrade a source link into verification."""
    scope = read_json(root/'docs/fact-check-scope.json')
    files, claims = [], []
    original_blobs=dict(line.split(' ',1)[::-1] for line in git(root,'ls-tree','-r','--format=%(objectname) %(path)',inventory['source_revision']).splitlines())
    nonclaims = {'.gitignore','.gitattributes','.python-version','LICENSE','.env.example'}
    additions={'docs/fact-check-scope.json','scripts/audit_saved_claims.py','scripts/build_publication.py','tests/test_saved_claims.py'}
    additions.update({'docs/publication/paper-halfcheetah.md.in','src/rl_locomotion_steering_vectors/publication/halfcheetah.py'})
    additions.update({'scripts/check_kinematics_followup.py','reports/fact-check-2026-10-29/kinematics-follow-up.md','reports/fact-check-2026-10-29/kinematics-follow-up.json'})
    additions.update(p.relative_to(root).as_posix() for p in (root/'src/rl_locomotion_steering_vectors/factcheck').glob('*.py'))
    for relative in sorted(set(inventory['original_publications'])|additions):
        path=root/relative
        groups=[g for g in scope['groups'] if any(fnmatch(relative,p) for p in g['locations'])]
        group=groups[0] if groups else None
        if path.suffix in ('.npz','.log') or relative.endswith(('results.json','manifest.json','selection.json','retarget.json')):
            disposition='preserved_observation_or_history'
        elif path.suffix in ('.png','.pdf','.mp4'):
            disposition='derived_visual_or_historical_media'
        elif relative in nonclaims:
            disposition='configuration_or_license_record'
        else:
            disposition=group['verdict'] if group else 'unresolved_evidence'
        operational=relative.startswith('reports/ant-publication/') and path.suffix not in ('.md','.png') and path.name not in ('evidence.json','inputs.json')
        files.append(dict(path=relative,disposition=disposition,groups=[g['id'] for g in groups],source_revision=inventory['source_revision'],
                          original_git_blob=original_blobs.get(relative),current_sha256=digest(path) if path.is_file() and not operational else None))
        if path.is_file() and path.suffix in ('.md','.in','.py'):
            for line,content in text_claims(path):
                if path.suffix != '.py' and content.startswith('# ') and '\n' not in content:
                    continue
                claims.append(dict(id=f'{relative}:{line}',location=dict(path=relative,line=line),claim=content,
                                   kind=group['kind'] if group else 'unclassified',verdict=disposition,
                                   independent_check=group['check'] if group else 'No independent factual validation; retained as an explicitly unresolved historical statement.',
                                   evidence=group['evidence'] if group else [],
                                   embedded_references=re.findall(r'https?://[^\s)>\]"`]+',content)))
    return dict(date=scope['date'],source_revision=inventory['source_revision'],interpretation=scope['interpretation'],
                groups=scope['groups'],files=files,claims=claims,primary_sources=scope['primary_sources'])


def summary_table(value):
    """Generate condition counts and frozen decisions from the independent verification."""
    lines=['| Experiment | Validation conditions | Confirmation conditions | Recorded outcome |','| --- | ---: | ---: | --- |']
    for name,e in value['experiments'].items():
        lines.append(f"| {name} | {len(e['phases'].get('validation',[]))} | {len(e['phases'].get('confirmation',[]))} | `{e['status']}` |")
    return '\n'.join(lines)


def report(value, registry, provenance):
    """Keep every empirical count bound to saved verification; interpretation remains explicit prose."""
    count=sum(len(rows) for e in value['experiments'].values() for rows in e['phases'].values())
    fresh=value['ant_audit']; p1=fresh['recomputed']['P1']
    statuses=Counter(x['verdict'] for x in registry['claims'])
    status_table='\n'.join(['| Claim-block disposition | Blocks |','| --- | ---: |']+[f'| {k} | {v} |' for k,v in sorted(statuses.items())])
    unmatched=sum(not x['source_commits'] for x in provenance['identities'].values())
    return f'''# Independent saved-data fact-check — 2026-10-29

Author: Libor “Burny” Burian. This audit starts from source revision `{value['source_revision']}` and treats earlier audits as claims, not authorities. It executes saved-array statistics and document tooling only. No policies, inference, gradients, training, simulator episodes or video replays were run.

## Numerical result and scope

The independent reference calculations reproduce all **{count} original validation/confirmation comparisons**. They also reconstruct **{len(fresh['arms'])} treated audit arms**, **{len(fresh['directions'])} signed activation directions**, and **{len(fresh['seeds'])} paired audit resets**. Means, paired intervals, failure counts, pilot gates, selected strengths, original fit/control vectors and later audit direction construction match within the documented tolerances. The later locked-primary mean is **{p1['mean']:+.6f} m/s**, BCa 95% interval **[{p1['bca_ci'][0]:.6f}, {p1['bca_ci'][1]:.6f}]**. These are arithmetic checks on saved observations, not new behavior measurements.

{summary_table(value)}

The [machine verification](verification.json) records every comparison. [Input hashes](inputs.json) preserve **{value['integrity']['files']} files**; none changed. Scalar comparisons use absolute tolerance {value['numerical_tolerance']['atol']:g} and relative tolerance {value['numerical_tolerance']['rtol']:g}. Reconstructed float32 vectors allow absolute error at most 1e-7; action-effort and planar-speed reductions allow 1e-10. Comparison ranks retain historical field names in immutable JSON; their interpretation is corrected below. Bootstrap procedures use the installed SciPy version recorded in [provenance](provenance.json), with explicit seeds and whole reset units.

## Corrections

- Random-direction and shuffled-control ranks are descriptive ensemble comparisons. Calibrated Monte Carlo p-values additionally require exchangeability under an explicit null, which norm matching and validation selection do not establish. Holm adjustment does not repair invalid input p-values. The historical arithmetic and decisions are preserved. [Assumptions](../../docs/methodology.md#statistical-assumptions-and-rank-interpretation).
- Odd/even components mean sign-dependent/sign-independent responses. They do not identify steering versus disruption. HalfCheetah slowdowns do not prove near-optimality or a unique degradation mechanism. Second-order optimality reasoning requires a local maximum of a smooth objective along the actual intervention path.
- “Conditions were confirmed” is replaced by “evaluated in confirmation.” Failed acceptance remains failed. Fixed fitted vectors are distinguished from observation-dependent activations.
- Hugging Face metadata no longer implies fine-tuning/merging of the frozen upstream policies. Explicit pinned checkpoint references remain. [Hub specification](https://huggingface.co/docs/hub/model-cards#specifying-a-base-model).
- Heading-relative kinematics pairs pre-transition input orientation with the ensuing displacement velocity. It is a discrete-time projection. Root-position velocity is not asserted to be bitwise identical to the torso-position velocity used by Ant reward.
- The [kinematics follow-up](kinematics-follow-up.md) corrects slope units for nondefault windows, single-window trend availability, angle branch-cut subtraction, and timestep validation. It checks that published default-window means and denominators remain compatible with saved evidence. A finite-horizon heading offset does not prove bounded long-run behavior.
- Further corrections distinguish a passing mean-effect gate from temporal persistence, selection bias from an observed validation/confirmation difference, covariance participation ratio from matrix rank, and current generated report bytes from historical run-directory renderings. Aligned directions can still be compared; alignment limits semantic interpretation rather than making an effect comparison untestable.
- Historical runtime checks, training metadata and paused PGS work are attributed records. Static inspection and saved arrays cannot establish untested model behavior. Existing perceived-heading claims retain the clipped-wrench qualification: rotating an already-clipped wrench need not equal clipping a physically rotated wrench.
- Archived README screenshots in `reports/readme-restructure` retain the former “p =” rank label; interpret it as the descriptive comparison rank above. Historical “perceived-heading axis” plot labels name the saved operational response axis, not a verified unique mediator or exact physical rotation. These preserved images do not supersede the corrected current prose and figures.

## Independent calculations

The reference implementation imports neither production estimators nor earlier audit calculations. It reads numeric NPZ members without pickle, verifies paired seeds/onset hashes/prefix arrays, reconstructs fitting windows and exclusions, episode-balanced activation RMS and contrasts, random streams, shuffle controls, split-half cosines, and frozen near-best selection. It retains prefix-only failures and unequal available horizons. Tests include missing/duplicate pairs, exact failure thresholds, within-window variance, unequal lengths, corrupt measurements and the reference coordinate transformation.

For the later audit, it recomputes P1/P2/S1/S2/M1/M2 arithmetic, including BCa/percentile intervals, sign-flip and Wilcoxon statistics, joint reset and crossed direction/reset resampling, ranks and Holm arithmetic. K1 geometry and its window denominators are recomputed. Original confirmation alternatives include prefix exclusion, both-arm survival, displacement, trimming, leave-one-out sensitivity and arm correlation. These alternatives are post hoc and do not replace the primary estimator. Conditional surviving horizons remain affected by treatment.

The publication check independently reconstructs all **{sum(x['conditions'] for x in value['publication']['cohorts'].values())} displayed conditions** and **{sum(x['trajectory_pairs'] for x in value['publication']['cohorts'].values())} trajectory pairs**, including reward-block denominators, failure markers, paired intervals and odd/even components. Display coordinates and reward blocks are rounded to six decimals before comparison; statistical effects retain full precision. HalfCheetah numerical prose and its complete locked-control table are generated from the independently checked saved results. Original report rendering uses explicit LF newlines so Git normalization cannot change deterministic bytes.

The saved neural response axes are used to check projections and component algebra; their neural derivation is not re-executed. The historical E1 fitted regression, power analyses, studentized sign-flip calculations and runtime replay results not explicitly present in verification.json remain **unresolved as independent reproductions**. They are not needed to retain the directly recomputed effect or failed acceptance. Earlier reports carry an erratum pointing here; their raw calculations remain preserved.

## Provenance and chronology

The provenance record maps cached runtime/perception fingerprints to historical Git source: **{unmatched} unmatched identities**. Checkpoint ZIP hashes and non-executable JSON training metadata were inspected without loading tensors or policies. Metadata records what was saved, not independently reproduced training history. Original policy parameter hashes, action-bias derivation and runtime cleanup remain historical evidence.

Original Ant's protocol first enters Git after diagnostics began. The later audit protocol's Git record precedes its logged collection; Git/log dates do not independently attest push time or prove that no earlier observations existed. HC retargeting deliberately reuses the earlier fitted height family. The audit overlaps **{len(provenance['audit_application_overlap'])} previously reserved application seeds**; it is not the formal replication split. Manifest decisions and failed/retried log events are retained in [provenance.json](provenance.json).

## Claim ledger and limits

The [claim ledger](claims.json) inventories **{len(registry['files'])} original and added source files** and **{len(registry['claims'])} prose/comment blocks**, linking locations, evidence, check scope and disposition. File coverage includes derived images/PDFs, recorded JSON, logs and source. Classification is conservative: a source citation does not automatically verify a claim; a group disposition applies only to its stated checks. Any uncovered block remains explicitly unresolved.

{status_table}

Primary-source reading and project adaptations are recorded in [the scope/source registry](../../docs/fact-check-scope.json) and [sources](../../docs/sources.md). The visual/media record and [checks actually executed](checks.json) are separate from old test totals. Movie metadata and saved correspondence receipts can bind artifacts but do not independently establish every frame's state correspondence without replay. No such replay was performed.

## Rebuild

Run `uv run --locked --no-sync python scripts/build_publication.py` to independently verify evidence and rebuild all current experiment reports, Ant figures/prose, the paper and this audit. Add `--check` to compare deterministic outputs without editing publications. The existing `scripts/build_ant_publication.py` remains available. Missing historical raw inputs must be restored using the pinned records; the checker never substitutes missing data or refreshes mismatching hashes.

Completion of these checks does not guarantee the absence of undiscovered errors. It establishes the documented dispositions and numerical agreements within the no-model boundary.
'''


def write_record(root,value,provenance,destination):
    """Write a deterministic ledger/report after corrections and publication rendering."""
    registry=ledger(root,read_json(root/OUT/'inputs.json'))
    outputs={OUT/'verification.json':encode(value),OUT/'provenance.json':encode(provenance),OUT/'claims.json':encode(registry),OUT/'report.md':report(value,registry,provenance).encode()}
    for relative,data in outputs.items():
        path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return [destination/p for p in outputs]
