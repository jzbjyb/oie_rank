from typing import List

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token

from rerank.dataset_readers.rerank_reader import SupOieConll


@Predictor.register('rerank')
class RerankPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict_conll_file(self, conll_filepath: str, batch_size: int = 256) -> List[float]:
        # collect instances
        insts = []
        soc = SupOieConll()
        for ext in soc.sentence_iterator(conll_filepath):
            verb_ind = [int(i == ext.pred_ind) for i in range(len(ext.words))]
            tokens = [Token(t) for t in ext.words]
            if not any(verb_ind):
                continue   # skip extractions without predicate
            insts.append(self._dataset_reader.text_to_instance(
                tokens, verb_ind, tags=soc.map_tags(ext.tags, one_verb=self._dataset_reader._one_verb)))

        # run rerank model and get the scores
        outputs = []
        for batch in range(0, len(insts), batch_size):
            batch = insts[batch:batch + batch_size]
            outputs.extend([p['scores'] for p in self._model.forward_on_instances(batch)])
        return outputs
