import argparse
import re
import logging

import sys
sys.path.append('.')


"""
This script checks whether the results format for Task 5 is correct. 
It also provides some warnings about possible errors.

The correct format of the Task 5 results file is the following:
<line_number> <TAB> <score>

where <line_number> is the number of the claim in the debate 
and <score> indicates the degree of 'check-worthiness' of the given line.
"""

_LINE_PATTERN_A = re.compile('^[1-9][0-9]{16,22}\t([-+]?\d*\.\d+|\d+)$')
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def check_format(file_path, subtask):
    if subtask == '1a':
        from format_checker.subtask_1a import check_format as cf
    elif subtask == '1b':
        from format_checker.subtask_1b import check_format as cf
    else:
        logging.error(f"Couldnt load the right format checker for the subtask-{subtask}")
        return False
    return cf(file_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-files-path", "-p", required=True, type=str, nargs="+",
                        help="The absolute pathes to the files you want to check.")
    parser.add_argument("--subtask", "-m", required=True, 
                        choices=['1a', '1b'],
                        help="The subtask you want to check the format of.")
    args = parser.parse_args()
    for pred_file_path in args.pred_files_path:
        logging.info(f"Subtask {args.subtask}: Checking file: {pred_file_path}")
        check_format(pred_file_path, args.subtask)