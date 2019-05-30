from typing import List, Optional

import torch, spacy, logging
import numpy as np

from allennlp.common.util import JsonDict, sanitize
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.open_information_extraction import make_oie_string, get_predicate_text
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

logger = logging.getLogger(__name__)


def n_best_viterbi_decode(tag_sequence: torch.Tensor,
                          transition_matrix: torch.Tensor,
                          tag_observations: Optional[List[int]] = None,
                          n_best: int = 1):
    '''
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep. Top n sequences with highest scores are decoded.
    :param tag_sequence:
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    :param transition_matrix:
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    :param tag_observations:
        A list of length ``sequence_length`` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.
    :param n_best:
        Keep n_best sequences with highest scores.
    :return:
        viterbi_path : torch.Tensor of shape (n_best, sequence_length)
            The tag indices of the top n tag sequences.
        viterbi_score : torch.Tensor of shape (n_best)
            The score of these sequences.
    '''
    sequence_length, num_tags = list(tag_sequence.size())
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError("Observations were provided, but they were not the same length "
                                     "as the sequence. Found sequence of length: {} and evidence: {}"
                                     .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    path_scores = []  # SHAPE: (seq_len * <=N * T)
    path_indices = []  # SHAPE: (seq_len * <=N * T)

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[:1, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        # SHAPE: (<=N, T, T)
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix.unsqueeze(0)
        # both SHAPE: (<=N, T)
        scores, paths = torch.topk(summed_potentials.view(-1, num_tags), n_best, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot.unsqueeze(0) + scores)
        else:
            path_scores.append(tag_sequence[timestep, :].unsqueeze(0) + scores)
        path_indices.append(paths)

    # Construct the top n most likely sequence backwards.
    # both SHAPE: (<=N)
    viterbi_score, ind = torch.topk(path_scores[-1].flatten(), n_best)
    ind_path = [ind]
    viterbi_path = [torch.remainder(ind, num_tags)]
    for backward_timestep in reversed(path_indices):
        ind = backward_timestep.flatten()[ind_path[-1]]
        ind_path.append(ind)
        viterbi_path.append(torch.remainder(ind, num_tags))
    # Reverse the backward path.
    viterbi_path.reverse()
    viterbi_path = torch.stack(viterbi_path, -1)
    return viterbi_path, viterbi_score


@Predictor.register('my-open-information-extraction')
class OpenIePredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True))
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "...", "predicate_index": "..."}``.
        Assumes sentence is tokenized, and that predicate_index points to a specific
        predicate (word index) within the sentence, for which to produce Open IE extractions.
        """
        tokens = json_dict["sentence"]
        predicate_index = int(json_dict["predicate_index"])
        verb_labels = [0 for _ in tokens]
        verb_labels[predicate_index] = 1
        return self._dataset_reader.text_to_instance(tokens, verb_labels)

    def _tag_tokens(self, tokens):
        tag = self.nlp.tagger(self.nlp.tokenizer.tokens_from_list(tokens))
        return tag

    def _beam_search(self, prob: np.ndarray, mask: np.ndarray, n_best: int = 1):
        log_prob = np.log(prob)
        seq_lens = mask.sum(-1).tolist()
        one_sam = False
        if log_prob.ndim == 2:
            one_sam = True
            p_li, lp_li, seq_lens = [prob], [log_prob], [seq_lens]
        else:
            p_li, lp_li, seq_lens = prob, log_prob, seq_lens
        all_tags, all_probs = [], []
        trans_mat = self._model.get_viterbi_pairwise_potentials()
        for p, lp, slen in zip(p_li, lp_li, seq_lens):
            # viterbi decoding (based on torch tensor)
            vpaths, vscores = n_best_viterbi_decode(
                torch.from_numpy(lp[:slen]), trans_mat, n_best=n_best)
            vpaths = vpaths.numpy()
            # collect tags and corresponding probs
            cur_tags, cur_probs = [], []
            for vpath in vpaths:
                probs = [p[i, vpath[i]] for i in range(len(vpath))]
                tags = [self._model.vocab.get_token_from_index(x, namespace='labels') for x in vpath]
                cur_probs.append(probs)
                cur_tags.append(tags)
            all_probs.append(cur_probs)
            all_tags.append(cur_tags)
        if one_sam:
            return all_tags[0], all_probs[0]
        return all_tags, all_probs

    def predict_batch(self,
                      sents: List[List[str]],
                      batch_size: int = 256,
                      warm_up: int = 0,
                      beam_search: int = 1) -> JsonDict:
        sents_token = [self._tag_tokens(sent) for sent in sents]

        instances, insts_st, insts_ed = [], [], []
        # find all verbs in the input sentence
        for sent_token in sents_token:
            pred_ids = [i for (i, t) in enumerate(sent_token) if t.pos_ == 'VERB']
            insts_st.append(len(instances))
            instances.extend([self._json_to_instance(
                {'sentence': sent_token, 'predicate_index': pid}) for pid in pred_ids])
            insts_ed.append(len(instances))

        # warm up the model using warm_up batch (mainly because of non-determinism of ELMo)
        if warm_up:
            for b in range(0, min(warm_up * batch_size, len(instances)), batch_size):
                batch_inst = instances[b:b + batch_size]
                self._model.forward_on_instances(batch_inst)

        # run model
        outputs, probs = [], []
        for b in range(0, len(instances), batch_size):
            batch_inst = instances[b:b+batch_size]
            for prediction in self._model.forward_on_instances(batch_inst):
                all_tags, all_probs = self._beam_search(
                    prediction['class_probabilities'], prediction['mask'], n_best=beam_search)
                outputs.append(all_tags)
                probs.append(all_probs)

        results_li = []
        for sent_token, st, ed in zip(sents_token, insts_st, insts_ed):
            # consolidate predictions
            cur_o = [e for o in outputs[st:ed] for e in o]
            cur_p = [e for o in probs[st:ed] for e in o]

            # Build and return output dictionary
            results = {'verbs': [], 'words': [token.text for token in sent_token]}

            for tags, prob in zip(cur_o, cur_p):
                # create description text
                description = make_oie_string(sent_token, tags)
                # add a predicate prediction to the return dictionary
                results['verbs'].append({
                    'verb': get_predicate_text(sent_token, tags),
                    'description': description,
                    'tags': tags,
                    'probs': prob,
                })
            results_li.append(results)

        return sanitize(results_li)
