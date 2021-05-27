import argparse
import re
import logging


"""
This script checks whether the results format for subtask-1a is correct. 
It also provides some warnings about possible errors.

The correct format of the subtask-1b results file is the following:
<line_number> <TAB> <score>

where <line_number> is the number of the claim in the debate 
and <score> indicates the degree of 'check-worthiness' of the given line.
"""

_LINE_PATTERN_A = re.compile('^[1-9][0-9]{0,3}\t([-+]?\d*\.\d+|\d+)$')
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def check_format(file_path):
    with open(file_path, encoding='UTF-8') as out:
        file_content = out.read().strip()
        for i, line in enumerate(file_content.split('\n')):
            if not _LINE_PATTERN_A.match(line.strip()):
                logging.error("Wrong line format: {}".format(line))
                return False

            line_number, score = line.split('\t')
            line_number = int(line_number)
            score = float(score.strip())

            if line_number != i + 1:
                logging.error('Problem with line_number: {}. They should be consecutive and starting from 1.'.format(line_number))
                return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-files-path", "-p", required=True, type=str, nargs='+',
                        help="The absolute pathes to the files you want to check.")
    args = parser.parse_args()
    for pred_file_path in args.pred_files_path:
        logging.info(f"Subtask 1B: Checking file: {pred_file_path}")
        check_format(pred_file_path)