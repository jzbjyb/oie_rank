import logging
from typing import Dict, List, Iterable

from overrides import overrides
import torch

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


class IntField(Field[int]):
    def __init__(self, value: int) -> None:
        self.value = value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.long)

    def __str__(self) -> str:
        return 'IntFeild'


class FloadField(Field[float]):
    def __init__(self, value: float) -> None:
        self.value = value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.float)

    def __str__(self) -> str:
        return 'FloatFeild'


class SupOieConllExtraction:
    def __init__(self,
                 words: List[str],
                 tags: List[str],
                 pred_ind: int,
                 label: int = None,
                 weight: float = None) -> None:
        self.words = words
        self.tags = tags
        self.pred_ind = pred_ind
        self.label = label
        self.weight = weight


class SupOieConll:
    def _conll_rows_to_extraction(self, rows):
        words = []
        tags = []
        pred_ind = None
        label = None
        weight = None
        for row in rows:
            row = row.split('\t')
            words.append(row[1])
            tags.append(row[7])
            pred_ind = int(row[4])
            label = None
            weight = None
            if len(row) > 8:
                label = int(row[8])
            if len(row) > 9:
                weight = float(row[9])
        return SupOieConllExtraction(words, tags, pred_ind, label, weight)

    def sentence_iterator(self, filepath: str) -> Iterable[SupOieConllExtraction]:
        with open(filepath, 'r') as fin:
            rows = []
            _ = fin.readline()  # skip csv head
            for l in fin:
                l = l.strip()
                if l == '':
                    if rows:
                        yield self._conll_rows_to_extraction(rows)
                        rows = []
                else:
                    rows.append(l)
            if rows:  # the last extraction
                yield self._conll_rows_to_extraction(rows)

    def map_tags_reverse(self, tags):
        ''' Map conll tags to sup-oie tags '''
        new_tags = []
        for tag in tags:
            if tag == 'O':
                new_tags.append(tag)
            else:
                bio, pa = tag.split('-')
                if bio not in {'B', 'I'}:
                    raise ValueError('tag error')
                if pa.startswith('ARG'):
                    pos = pa[3:]
                    new_tags.append('A{}-{}'.format(int(pos), bio))
                elif pa =='V':
                    new_tags.append('P-B')
                else:
                    new_tags.append('O')
        return new_tags

    def map_tags(self, tags, one_verb=True):
        ''' Map sup-oie tags to conll tags '''
        new_tags = []
        for tag in tags:
            nv = 0
            if tag == 'O':
                new_tags.append(tag)
            else:
                name, bio = tag.split('-')
                if bio not in {'B', 'I'}:
                    raise ValueError('tag error')
                if name.startswith('A'):
                    pos = name[1:]
                    new_tags.append('{}-ARG{}'.format(bio, int(pos)))
                elif name == 'P':
                    if one_verb and (bio != 'B' or nv > 0):
                        new_tags.append('O')
                    else:
                        nv += 1
                        new_tags.append('{}-{}'.format(bio, 'V'))
                else:
                    raise ValueError('tag error')
        return new_tags


@DatasetReader.register('rerank')
class RerankReader(DatasetReader):
    '''
    Read conll format used in sup-oie project (https://github.com/gabrielStanovsky/supervised-oie).
    Note that the conll format used in sup-oie is slightly different from standard conll format.
    '''
    def __init__(self,
                 one_verb: bool = True,  # if True, at most one "V" tag are included for each extraction
                 skip_neg: bool = False,  # if True, negative samples are skipped
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._one_verb = one_verb
        self._skip_neg = skip_neg
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, filepath: str):
        filepath = cached_path(filepath)
        soc = SupOieConll()
        for ext in soc.sentence_iterator(filepath):
            verb_ind = [int(i == ext.pred_ind) for i in range(len(ext.words))]
            tokens = [Token(t) for t in ext.words]
            if not any(verb_ind):
                continue  # skip extractions without predicate
            if self._skip_neg and ext.label == 0:
                continue  # skip negative examples
            yield self.text_to_instance(
                tokens, verb_ind, tags=soc.map_tags(ext.tags, one_verb=self._one_verb),
                label=ext.label, weight=ext.weight)

    def text_to_instance(self,
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str],
                         task: str = None,
                         label: int = None,
                         weight: float = None) -> Instance:
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        fields['tags'] = SequenceLabelField(tags, text_field)
        if label is not None:
            fields['labels'] = IntField(label if label != 0 else -1)
        if weight is not None:
            fields['weights'] = FloadField(weight)
        if not any(verb_label):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        fields['metadata'] = MetadataField({
            'words': [x.text for x in tokens],
            'verb': verb})
        return Instance(fields)
