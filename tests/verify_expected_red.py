"""Remote-only RED gate: expected assertion failures are evidence, not a broken CI job."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

files = ['tests/test_native_pipeline.py', 'tests/test_native_evaluation.py',
         'tests/test_training_overhead.py', 'tests/test_compute_runtime.py']
report = Path('expected-red.xml')
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *files,
                         f'--junitxml={report}', '--tb=short'])
if result.returncode != 1:
    raise SystemExit('Expected failing assertions against the old implementation; got exit '
                     + str(result.returncode))
root = ET.parse(report).getroot()
cases = root.findall('.//testcase')
if root.findall('.//error'):
    raise SystemExit('Collection/environment errors do not count as expected RED')
failed = [case for case in cases if case.find('failure') is not None]
classes = {case.attrib.get('classname', '') for case in failed}
for name in ('test_native_pipeline', 'test_native_evaluation', 'test_training_overhead',
             'test_compute_runtime'):
    if not any(name in classname for classname in classes):
        raise SystemExit('No demonstrated failure for ' + name)
print(f'EXPECTED RED VERIFIED: {len(failed)} assertions fail against the unchanged production code.')
