import pdb
import logging
import argparse
import os

import sys
sys.path.append('.')
from format_checker.main import check_format
from scorer.utils import _compute_average_precision, _compute_reciprocal_rank, _compute_precisions
from scorer.utils import print_thresholded_metric, print_single_metric
"""
Scoring of Task 5 with the metrics Average Precision, R-Precision, P@N, RR@N. 
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50]

def _read_gold_and_pred(gold_fpath, pred_fpath):
    """
    Read gold and predicted data.
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number and score at each line.
    :return: {line_number:label} dict; list with (line_number, score) tuples.
    """

    logging.info("Reading gold predictions from file {}".format(gold_fpath))

    gold_labels = {}
    with open(gold_fpath, encoding='utf-8') as gold_f:

        for line_res in gold_f:
            (topic_id, tweet_id, tweet_url, tweet_text, check_worthiness) = line_res.strip().split('\t')  # process the line from the res file
            if topic_id == 'topic_id':
                continue
            label = check_worthiness
            gold_labels[int(tweet_id)] = int(label)

    logging.info('Reading predicted ranking order from file {}'.format(pred_fpath))

    line_score = []
    with open(pred_fpath) as pred_f:
        for line in pred_f:
            topic_id, tweet_id, score, run_id  = line.split('\t')
            tweet_id = int(tweet_id.strip())
            score = float(score.strip())

            if tweet_id not in gold_labels:
                logging.error('No such tweet_id: {} in gold file!'.format(tweet_id))
                quit()
            line_score.append((tweet_id, score))

    if len(set(gold_labels).difference([tup[0] for tup in line_score])) != 0:
        logging.error('The predictions do not match the lines from the gold file - missing or extra line_no')
        raise ValueError('The predictions do not match the lines from the gold file - missing or extra line_no')

    return gold_labels, line_score


def evaluate(gold_fpath, pred_fpath, thresholds=None):
    """
    Evaluates the predicted line rankings w.r.t. a gold file.
    Metrics are: Average Precision, R-Pr, Reciprocal Rank, Precision@N
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number at each line, where the list is ordered by check-worthiness.
    :param thresholds: thresholds used for Reciprocal Rank@N and Precision@N.
    If not specified - 1, 3, 5, 10, 20, 50, len(ranked_lines).
    """
    gold_labels, line_score = _read_gold_and_pred(gold_fpath, pred_fpath)

    ranked_lines = [t[0] for t in sorted(line_score, key=lambda x: x[1], reverse=True)]
    if thresholds is None or len(thresholds) == 0:
        thresholds = MAIN_THRESHOLDS + [len(ranked_lines)]

    # Calculate Metrics
    precisions = _compute_precisions(gold_labels, ranked_lines, len(ranked_lines))
    avg_precision = _compute_average_precision(gold_labels, ranked_lines)
    reciprocal_rank = _compute_reciprocal_rank(gold_labels, ranked_lines)
    num_relevant = len({k for k, v in gold_labels.items() if v == 1})

    return thresholds, precisions, avg_precision, reciprocal_rank, num_relevant


def validate_files(pred_file, subtask):
    if not check_format(pred_file, subtask):
        logging.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file-path", "-g", required=True, type=str,
                        help="Path to files with gold annotations.")
    
    parser.add_argument("--pred-file-path", "-p", required=True, type=str,
                        help="Path to files with ranked line_numbers.")

    args = parser.parse_args()

    line_separator = '=' * 120
    pred_file = args.pred_file_path
    gold_file = args.gold_file_path
    subtask = '1a'

    if validate_files(pred_file, subtask):
        logging.info(f"Started evaluating results for subtask-{subtask} ...")
        overall_precisions = [0.0] * len(MAIN_THRESHOLDS)

        thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate(gold_file, pred_file)
        threshold_precisions = [precisions[th - 1] for th in MAIN_THRESHOLDS]
        r_precision = precisions[num_relevant - 1]

        for idx in range(0, len(MAIN_THRESHOLDS)):
            overall_precisions[idx] = threshold_precisions[idx]
        mean_r_precision = r_precision
        mean_avg_precision = avg_precision
        mean_reciprocal_rank = reciprocal_rank

        filename = os.path.basename(pred_file)
        logging.info('{:=^120}'.format(' RESULTS for {} '.format(filename)))
        print_single_metric('AVERAGE PRECISION:', avg_precision)
        print_single_metric('RECIPROCAL RANK:', reciprocal_rank)
        print_single_metric('R-PRECISION (R={}):'.format(num_relevant), r_precision)
        print_thresholded_metric('PRECISION@N:', MAIN_THRESHOLDS, threshold_precisions)

