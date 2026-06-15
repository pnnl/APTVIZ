import sys
import os
import types
import importlib.util
from pathlib import Path

# Bootstrap: register this directory as a Python package so that relative
# imports inside ccd.py (e.g. `from . import unpack`) work when running
# this script directly instead of via `python -m`.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_name = 'apt_ccd'
_pkg = types.ModuleType(_pkg_name)
_pkg.__path__ = [_script_dir]
_pkg.__package__ = _pkg_name
sys.modules[_pkg_name] = _pkg

_ccd_spec = importlib.util.spec_from_file_location(
    f'{_pkg_name}.ccd', os.path.join(_script_dir, 'ccd.py')
)
ccd = importlib.util.module_from_spec(_ccd_spec)
ccd.__package__ = _pkg_name
sys.modules[f'{_pkg_name}.ccd'] = ccd
_ccd_spec.loader.exec_module(ccd)

import json
import argparse

parser = argparse.ArgumentParser(description='Preprocess APT data to generate neighborhood metadata.')
parser.add_argument('--rrng', type=str, required=True, help='Path to the RRNG file')
parser.add_argument('--data', type=str, required=True, help='Path to the APT data file')
parser.add_argument('--savedir', type=str, required=True, help='Directory to save the output')
args = parser.parse_args()

# Convert .apt file to .pos
if Path(args.data).suffix == '.apt':
    import apav as ap
    roi = ap.load_apt(args.data)
    args.data = args.data.replace('.apt','.pos')
    roi.to_pos(args.data)

# Generate neighborhoods and metadata
out = ccd.generate_neighborhoods(args.data, args.rrng, savedir=args.savedir)

with open(f'{args.savedir}/{Path(args.data).stem}_metadata.json', 'w') as json_file:
    json.dump(out, json_file, indent=4) 
