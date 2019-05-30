import argparse

from allennlp.common.util import import_submodules
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='re-calculate confidence scores')
    parser.add_argument('--model', type=str, help='model file', required=True)
    parser.add_argument('--inp', type=str, help='input sup-oie conll and extraction file with the same order',
                        required=True)
    parser.add_argument('--out', type=str, help='output file.', required=True)
    parser.add_argument('--cuda_device', type=int, default=0, help='id of GPU to use (if any)')
    args = parser.parse_args()

    import_submodules('rerank')

    arc = load_archive(args.model, cuda_device=args.cuda_device)
    predictor = Predictor.from_archive(arc, predictor_name='rerank')
    conll_file, ext_file = args.inp.split(':')
    scores = predictor.predict_conll_file(conll_file, batch_size=256)
    count = 0
    with open(ext_file, 'r') as fin, open(args.out, 'w') as fout:
        for i, l in enumerate(fin):
            l = l.split('\t')
            l[1] = str(scores[i])
            fout.write('\t'.join(l))
            count += 1
    assert count == len(scores), 'number of scores is not equal to number of extractions'
