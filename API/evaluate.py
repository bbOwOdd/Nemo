import argparse
import json
import re
import string
from collections import Counter
from rouge_score import rouge_scorer

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--pred_file',
        type=str,
        help="Text file with test set prompts + model predictions. Prediction file can be made by running NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning_eval.py",
    )
    parser.add_argument(
        '--pred_field',
        type=str,
        help="The field in the json file that contains the prediction tokens",
        default="pred",
    )
    parser.add_argument(
        '--label_field',
        type=str,
        help="The field in the json file that contains the ground truth tokens",
        default="label",
    )

    args = parser.parse_args()

    pred_file = args.pred_file
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    preds = open(pred_file, encoding="utf-8").readlines()
    f1 = exact_match = total = r_score = 0

    for i in range(len(preds)):
        pred_line = json.loads(preds[i])

        pred_answer = pred_line[args.pred_field]
        true_answers = pred_line[args.label_field]
        if not isinstance(true_answers, list):
            true_answers = [true_answers]

        r_scores = []
        for ta in true_answers:
            r_scores.append(scorer.score(ta, pred_answer)['rougeL'].fmeasure)
        r_score += max(r_scores)
        exact_match += metric_max_over_ground_truths(exact_match_score, pred_answer, true_answers)
        f1 += metric_max_over_ground_truths(f1_score, pred_answer, true_answers)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    r_score = 100 * (r_score / total)
    res = {'exact_match': exact_match, 'f1': f1, "rougeL": r_score, 'total': total}
    print('\t'.join([f"{k} {v:.3f}" for k, v in res.items()]))


if __name__ == "__main__":
    main()
