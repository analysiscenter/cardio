""" Simple script to format PhysioNet/CinC 2017 Challenge data according to
wfdb standard."""

import os
import argparse
import re


def check_format(line_values):

    new_time_fmt = re.compile(r'\d{2}:\d{2}:\d{2}')
    new_date_fmt = re.compile(r'\d{2}/\d{2}/\d{4}')

    old_time_fmt = re.compile(r'\d{2}:\d{2}:\d{2}')
    old_date_fmt = re.compile(r'\d{4}-\d{2}-\d{2}')

    if ((new_time_fmt.fullmatch(line_values[4]) is not None) and 
        (new_date_fmt.fullmatch(line_values[5]) is not None)):
        which_format = 'new'
    elif ((old_time_fmt.fullmatch(line_values[5]) is not None) and 
          (old_date_fmt.fullmatch(line_values[4]) is not None)):
        which_format = 'old'
    else:
        which_format = None
    return which_format

def main():

    parser = argparse.ArgumentParser(description='Path to the files and mask')
    parser.add_argument('-p', '--path', help='Full path to PN2017 files')

    args = parser.parse_args()

    all_files = os.listdir(args.path)
    header_files = [file for file in all_files if file.endswith(".hea")]

    for header in header_files:
        path = os.path.join(args.path, header)
        with open(path, 'r') as f:
            lines = f.readlines()
        # We have lines[0]: 'A00005 1 300 18000 2013-12-23 08:12:08 \n'
        line_values = lines[0].split(' ')
        if len(line_values) == 7:
            which_format = check_format(line_values)
            if which_format == 'old':
                new_order = [0, 1, 2, 3, 5, 4, 6]
                line_values[4] = '/'.join(line_values[4].split('-')[::-1])
                lines[0] = ' '.join([line_values[i] for i in new_order])
                # Now we get lines[0]: 'A00005 1 300 18000 08:12:08 23/12/2013 \n'
            elif which_format != 'new':
                raise ValueError('Unexpected header format in record {}.'.format(header))
        elif len(line_values) < 7:
        # e.g. in case lines[0]: 'A00005 1 300 18000 2013-12-23 \n'
            new_order = [0, 1, 2, 3, -1]
            lines[0] = ' '.join([line_values[i] for i in new_order])
            # lines[0]: 'A00005 1 300 18000 \n'
        else:
            raise ValueError('Unexpected header format in record {}.'.format(header))
        with open(path, 'w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    main()