#!/usr/bin/env python3
"""Check synthetic two-label, two-round bandwidth against hand calculations."""
import pathlib
import subprocess
import sys

fixtures = pathlib.Path(__file__).resolve().parent
root = fixtures.parents[2]
command = [sys.executable, str(root / 'bin/mm-gbps.py'),
           *(f'{fixtures / label}-r1.sql,{fixtures / label}-r2.sql' for label in ('a', 'b')),
           '--labels', 'upstream,mmvh']

# Q4_K 4096x4096: weights = 65536 blocks * 144 = 9437184 bytes;
# n=8: +131072 activations +131072 output = 9699328 bytes;
# n=16: +262144 activations +262144 output = 9961472 bytes.
# Q8_0 1024x4096: weights = 131072 blocks * 34 = 4456448 bytes;
# n=8: +131072 activations +32768 output = 4620288 bytes;
# n=16: +262144 activations +65536 output = 4784128 bytes.
# Medians: upstream (100,200) us, mmvh (50,100) us. Decimal GB/s
# is bytes / (us * 1000); percentages below divide GB/s by 732 or 100.
expected = {
    ('Q4_K', '4096', '4096', 'upstream'): ['100.000', '96.993', '13.25%', '200.000', '49.807', '6.80%'],
    ('Q4_K', '4096', '4096', 'mmvh'): ['50.000', '193.987', '26.50%', '100.000', '99.615', '13.61%'],
    ('Q8_0', '1024', '4096', 'upstream'): ['100.000', '46.203', '6.31%', '200.000', '23.921', '3.27%'],
    ('Q8_0', '1024', '4096', 'mmvh'): ['50.000', '92.406', '12.62%', '100.000', '47.841', '6.54%'],
}
custom_percentages = {
    ('Q4_K', '4096', '4096', 'upstream'): ('96.99%', '49.81%'),
    ('Q4_K', '4096', '4096', 'mmvh'): ('193.99%', '99.61%'),
    ('Q8_0', '1024', '4096', 'upstream'): ('46.20%', '23.92%'),
    ('Q8_0', '1024', '4096', 'mmvh'): ('92.41%', '47.84%'),
}
for extra, peak in [((), '732'), (('--peak-gbps', '100'), '100')]:
    output = subprocess.check_output([*command, *extra], text=True)
    assert f'Roofline: {peak} GB/s' in output, output
    assert '| n=8 µs | n=8 GB/s | n=8 % peak | n=16 µs | n=16 GB/s | n=16 % peak |' in output
    rows = {}
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if cells[0] in ('Q4_K', 'Q8_0'):
            key = tuple(cells[:4])
            assert key not in rows, key
            rows[key] = cells[4:]
    wanted = {key: list(cells) for key, cells in expected.items()}
    if extra:
        for key, (p8, p16) in custom_percentages.items():
            wanted[key][2], wanted[key][5] = p8, p16
    assert rows == wanted, (rows, wanted)
for extra in [('--labels', 'only-one'), ('--peak-gbps', '0'), ('--peak-gbps', 'nan')]:
    result = subprocess.run([*command, *extra], text=True, capture_output=True)
    assert result.returncode != 0, extra
print('mm_gbps: hand-calculated GB/s, medians, labels and rooflines passed')
