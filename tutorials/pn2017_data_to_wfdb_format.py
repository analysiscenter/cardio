""" Simple script to format PhysioNet/CinC 2017 Challenge data according to wfdb standard."""

import os
import argparse

parser = argparse.ArgumentParser(description='Path to the files and mask')
parser.add_argument('--path', type=str, dest='path', default='./', 
                    help='Full path to PN2017 files')

args = parser.parse_args()

header_files = [file for file in os.listdir(args.path) if file.endswith(".hea")]

for header in header_files:
    path = os.path.join(args.path, header)
    with open(path, 'r') as f:
        lines = f.readlines()
    # lines[0]: 'A00005 1 300 18000 2013-12-23 08:12:08 \n'
    line_values = lines[0].split(' ')
    try:
        new_order = [0, 1, 2, 3, 5, 4, 6]
        line_values[4] = '/'.join(line_values[4].split('-')[::-1])
        lines[0] = ' '.join([line_values[i] for i in new_order])
        # lines[0]: 'A00005 1 300 18000 08:12:08 23/12/2013 \n'
    except IndexError:
    # in case lines[0]: 'A00005 1 300 18000 2013-12-23 \n'
        new_order = [0, 1, 2, 3, -1]
        lines[0] = ' '.join([line_values[i] for i in new_order])
        # lines[0]: 'A00005 1 300 18000 \n'
    with open(path, 'w') as f:
        f.writelines(lines)
