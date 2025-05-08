from rouge_score import rouge_scorer
import argparse
import json
import os


def calculate_rouge(reference: str, generated: str, output: dict) -> dict:
    """
    Calculate the ROUGE score of a sample given reference and generated text.
    :param reference: A string containing the reference text.
    :param generated: A string containing the generated text.
    :param output: A dict containing scores for previous samples.
    :return: The input dict with the scores for the current sample appended.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    idx = len(output)+1

    output[idx] = scores
    return output


def get_macro_scores(output: dict) -> dict:
    """
    Averages the ROUGE scores across all entries in the dict.
    :param output: A dict containing scores for all evaluated samples.
    :return: A dict with the average scores.
    """
    macros = {'rouge1': [0, 0, 0], 'rouge2': [0, 0, 0], 'rougeL': [0, 0, 0]}

    for scores in output.values():
        macros['rouge1'] = [x + y for x, y in zip(macros['rouge1'], scores['rouge1'])]
        macros['rouge2'] = [x + y for x, y in zip(macros['rouge2'], scores['rouge2'])]
        macros['rougeL'] = [x + y for x, y in zip(macros['rougeL'], scores['rougeL'])]

    macros['rouge1'] = [x / 20 for x in macros['rouge1']]
    macros['rouge2'] = [x / 20 for x in macros['rouge2']]
    macros['rougeL'] = [x / 20 for x in macros['rougeL']]

    return macros


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evaluate.py',
        description='Evaluate RAG output using ROUGE.'
    )
    parser.add_argument(
        'reference_txt',
        type=str,
        help='A file containing the reference text, which is the vector search result.'
    )
    parser.add_argument(
        'generated_txt',
        type=str,
        help='A file containing the generated text, which is the LLM output.'
    )
    parser.add_argument(
        'results',
        type=str,
        help='The json file containing the evaluation results.'
    )
    args = parser.parse_args()

    with open(args.reference_txt, 'r', encoding='utf-8') as f:
        reference = f.readlines()
        reference = ''.join(reference)

    with open(args.generated_txt, 'r', encoding='utf-8') as f:
        generated = f.readlines()
        generated = ''.join(generated)

    with open(args.results, 'r', encoding='utf-8') as f:
        results_dict = json.load(f)

    with open(args.results, 'w', encoding='utf-8') as f:
        results = calculate_rouge(reference, generated, results_dict)
        json.dump(results, f, ensure_ascii=False, indent=2)

    macros = get_macro_scores(results_dict)

    macro_txt = os.path.join(os.path.dirname(args.results), 'macro_scores.txt')

    with open(macro_txt, 'w', encoding='utf-8') as f:
        for key, macro in macros.items():
            f.write(f'{key}\n')
            f.write(f'\tprecision: {macro[0]}\n')
            f.write(f'\trecall: {macro[1]}\n')
            f.write(f'\tf-score: {macro[2]}\n')
