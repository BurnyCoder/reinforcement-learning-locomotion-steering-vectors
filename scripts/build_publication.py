"""Independently check saved research and rebuild every current publication, without models.

Usage: uv run --locked --no-sync python scripts/build_publication.py [--check]
The original Ant builder remains available. This wrapper adds all experiments and
the independent dated claim record; raw evidence and original selections are read-only.
"""
import argparse
import importlib.util
import logging
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_saved_claims import generate as audit
from rl_locomotion_steering_vectors.factcheck.audit import EXPERIMENTS
from rl_locomotion_steering_vectors.factcheck.provenance import provenance
from rl_locomotion_steering_vectors.factcheck.display import publication
from rl_locomotion_steering_vectors.factcheck.record import OUT, write_record
from rl_locomotion_steering_vectors.publication.inputs import validate_inputs
from rl_locomotion_steering_vectors.storage import start_logging

NOTICE=("This is a saved-data rendering of the original experiment, not a new run. Original JSON, selections and failed outcomes are unchanged. "
        "Historical decisions and runtime checks below are attributed records. The independent 2026-10-29 audit recomputes original extraction, paired effects, intervals, failures and gates. "
        "Odd/even responses do not identify a unique mechanism; random-direction ranks need a justified exchangeability null for p-value interpretation. "
        "Current corrections and verification limits: https://github.com/BurnyCoder/reinforcement-learning-locomotion-steering-vectors/blob/main/reports/fact-check-2026-10-29/report.md")


def render_experiments(root, destination):
    """Keep the same evidence-link context in a temporary and a normal report build."""
    from reportlab import rl_config
    from rl_locomotion_steering_vectors.reporting import write_report
    rl_config.invariant=1
    outputs=[]
    for name in EXPERIMENTS:
        folder=destination/'reports'/name
        folder.mkdir(parents=True,exist_ok=True)
        if destination != root:
            for filename in ('manifest.json','results.json','vector_diagnostics.json'):
                source = root/'reports'/name/filename
                if source.is_file():
                    shutil.copyfile(source,folder/filename)
        write_report(folder,timestamp='2026-10-29 (saved-data publication)',notice=NOTICE)
        outputs.extend([folder/'report.md',folder/'report.pdf',*sorted(folder.glob('strength_response_*.png'))])
    return outputs


def build(root, destination, value, source_record):
    """Reuse renderers while keeping all inputs outside the generated-output list."""
    from scripts.build_ant_publication import generate as ant
    outputs = render_experiments(root, destination)
    from scripts.check_kinematics_followup import generate as kinematics_followup
    outputs.extend(kinematics_followup(root, destination))
    inputs=validate_inputs(root,root/'reports/ant-publication/inputs.json',restore=False)
    outputs.extend(ant(root,inputs,destination))
    value['publication'] = publication(root, destination)
    if value['publication']['differences']:
        raise ValueError(f"Generated publication disagrees with saved arrays: {value['publication']['differences']}")
    outputs.extend(write_record(root,value,source_record,destination))
    spec=importlib.util.spec_from_file_location('factcheck_pdf',root/'reports/build_paper.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    outputs.append(renderer.build(destination/OUT/'report.md'))
    return outputs


def main():
    """Recompute independently before invoking production renderers, then compare every output byte."""
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    start_logging(ROOT/'tmp/full-publication')
    value=audit(ROOT,check=args.check,include_publication=False)
    mismatches=[d for e in value['experiments'].values() for d in e['differences']+e['fitting'].get('differences',[])]+value['ant_audit']['differences']
    if mismatches:
        raise ValueError(f'Independent arithmetic disagrees: {mismatches}')
    source_record=provenance(ROOT)
    if args.check:
        with tempfile.TemporaryDirectory(prefix='publication-',dir=ROOT/'tmp') as temp:
            stage=Path(temp)
            outputs=build(ROOT,stage,value,source_record)
            stale=[p.relative_to(stage).as_posix() for p in outputs if not (ROOT/p.relative_to(stage)).is_file() or p.read_bytes()!=(ROOT/p.relative_to(stage)).read_bytes()]
        if stale:
            logging.error('Stale outputs: %s',stale);return 1
        logging.info('Verified %d deterministic publication outputs',len(outputs))
    else:
        outputs=build(ROOT,ROOT,value,source_record)
        logging.info('Built %d publication outputs; no models or simulation',len(outputs))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
