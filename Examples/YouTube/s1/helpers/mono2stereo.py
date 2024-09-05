# convert audio file from mono-channel to stereo-channel
# 
# reference: https://stackoverflow.com/questions/43162121/python-convert-mono-wave-file-to-stereo
# 
# Zhenhao Ge, 2024-07-28 

import os, sys
import getopt
from pathlib import Path

# set dirs
home_dir = str(Path.home())
work_dir = os.path.join(home_dir, 'code', 'repo', 'style-tts2')
if os.getcwd() != work_dir:
    os.chdir(work_dir)
print('current dir: {}'.format(os.getcwd()))

from audio import make_stereo

argv = sys.argv[1:]
try:
    options, args = getopt.getopt(argv, "i:o:", ["infile=", "outfile="])
    print(options)
except:
    print("mono2stereo.py --infile <infile> --outfile <outfile>")

for name, value in options:
    # print(f'name: {name}, value: {value}')
    if name in ['-i', '--infile']:
        infile = value
    elif name in ['-o', '--outfile']:
        outfile = value

print(f'input file: {infile}')
print(f'output file: {outfile}')

if __name__ == '__main__':

    make_stereo(infile, outfile)