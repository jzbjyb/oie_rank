from typing import List, Dict, Union

import argparse
import itertools
import numpy as np

from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules, JsonDict


def read_sents(filepath: str, format='raw') -> List[Union[List[str], JsonDict]]:
    sents_token = []
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l == '':
                continue
            sents_token.append(l.split(' '))
    if format == 'raw':
        return sents_token
    if format == 'json':
        return [{'sentence': sent} for sent in sents_token]


class Extraction:
    def __init__(self, sent, pred, args, confidence):
        self.sent = sent
        self.pred = pred
        self.args = args
        self.confidence = confidence

    def _format_comp(self, comp):
        return ' '.join(map(lambda x: x[0], comp)) + '##' + str(comp[0][1])

    def __str__(self):
        return '\t'.join(map(str, [' '.join(self.sent), self.confidence, self._format_comp(self.pred),
                                   '\t'.join([self._format_comp(arg) for arg in self.args])]))


def prediction_to_extraction(preds: List[Dict],
                             max_n_arg: int = 10,
                             keep_one: bool = False,
                             merge: bool = True) -> List[Extraction]:
    '''
    Generate extractions from tag sequences. Assume tag sequences are valid (no sanity check).
    :param preds:
        Predictions output by open IE models.
    :param max_n_arg:
        Maximum number of arguments allowed.
    :param keep_one:
        If True, only keep the first instance for each argument.
        E.g., if two ARG0s are labeled, only keep the first one.
    :param merge:
        If False, instances of the same argument will be separated into different extractions.
        E.g., if two ARG0s are labeled, two extractions are generated for them respectively.
    :return:
        List of extractions.
    '''
    print('keep_one: {}, merge: {}'.format(keep_one, merge))
    exts = []
    n_trunc_ext = 0  # number of extractions with more than one instances identified for a specific ARG
    n_more_pred = 0  # number of extractions with more than one predicate identified
    calc_conf = lambda x: np.mean(np.log(np.clip(x, 1e-5, 1 - 1e-5)))  # use avg log prob as confidence
    for pred in preds:
        tokens = pred['words']
        for ext in pred['verbs']:
            probs = []
            pred = []
            args = [[] for _ in range(max_n_arg)]
            last_is = -1  # -1 for start and O, -2 for V, others for ARG
            for i, w, t, p in zip(range(len(tokens)), tokens, ext['tags'], ext['probs']):
                probs.append(p)
                if t.find('V') >= 0:
                    if last_is != -2:
                        pred.append([])
                    pred[-1].append((w, i))
                    last_is = -2
                elif t.find('ARG') >= 0:
                    ai = int(t[t.find('ARG')+3:])
                    if ai >= len(args):
                        raise ValueError('too many ARGs')
                    if ai < 0:
                        raise ValueError('negative ARG position')
                    if last_is != ai:
                        args[ai].append([])  # create new ARG placeholder
                    args[ai][-1].append((w, i))
                    last_is = ai
                else:
                    last_is = -1
            # remove empty ARG position (e.g., ARG2 exists without ARG1)
            args = [arg for arg in args if len(arg) > 0]
            # skip tag sequences without predicates or arguments
            if len(pred) <= 0 or len(args) <= 0:
                continue
            # only keep the first predicate
            if len(pred) > 1:
                n_more_pred += 1
            pred = pred[0]
            # only keep the first instance of each ARG (should be done before merge)
            if keep_one:
                n_trunc_ext += any([len(arg) > 1 for arg in args])
                args = [arg[:1] for arg in args]
            # merge all instances of an ARG
            if merge:
                args = [[[w for a in arg for w in a]] for arg in args]
            # iterate through all combinations
            for arg in itertools.product(*args):
                exts.append(Extraction(sent=tokens, pred=pred, args=arg, confidence=calc_conf(probs)))
    print('{} extractions are truncated, {} extractions have more than one predicates'.format(
        n_trunc_ext, n_more_pred))
    return exts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract open IE extractions')
    parser.add_argument('--model', type=str, help='model file', required=True)
    parser.add_argument('--inp', type=str, help='input file of raw sentences', required=True)
    parser.add_argument('--out', type=str, help='output file', required=True)
    parser.add_argument('--cuda_device', type=int, default=0, help='id of GPU to use (if any)')
    parser.add_argument('--beam_search', type=int, default=1, help='beam size')
    parser.add_argument('--unmerge', help='whether to generate multiple extractions for a predicate',
                        action='store_true')
    parser.add_argument('--keep_one', help='whether to keep only the first instance for each ARG',
                        action='store_true')
    args = parser.parse_args()

    import_submodules('rerank')

    # run oie mode to get extractions
    arc = load_archive(args.model, cuda_device=args.cuda_device)
    predictor = Predictor.from_archive(arc, predictor_name='my-open-information-extraction')
    sents_tokens = read_sents(args.inp, format='raw')
    preds = predictor.predict_batch(sents_tokens, batch_size=256, warm_up=3, beam_search=args.beam_search)

    # convert tag sequences to extractions
    exts = prediction_to_extraction(preds, max_n_arg=10, merge=not args.unmerge, keep_one=args.keep_one)

    # save extractions
    with open(args.out, 'w') as fout:
        for ext in exts:
            fout.write('{}\n'.format(ext))
