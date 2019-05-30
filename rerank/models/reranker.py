from typing import Dict, List, Optional, Any

from overrides import overrides
import torch
from torch.nn.functional import margin_ranking_loss

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.models.semantic_role_labeler import SemanticRoleLabeler
from allennlp.nn import InitializerApplicator, RegularizerApplicator


@Model.register('reranker')
class Reranker(SemanticRoleLabeler):
    '''
    Rank-aware Open IE model that calibrates the probability of an extraction for better ranking performance.
    '''
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False) -> None:
        super().__init__(vocab, text_field_embedder, encoder, binary_feature_dim, embedding_dropout,
                         initializer, regularizer, label_smoothing, ignore_span_metric)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                tags: torch.LongTensor,
                labels: torch.LongTensor = None,  # binary label of each extraction
                weights: torch.LongTensor = None,  # weight of each extraction
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # run open IE model to get extraction likelihood
        output_dict = super().forward(tokens, verb_indicator, metadata=metadata)
        # SHAPE: (batch_size, seq_len, num_classes)
        probs = output_dict['class_probabilities']
        # SHAPE: (batch_size, seq_len)
        mask = output_dict['mask']
        probs = torch.clamp(probs, 1e-5, 1 - 1e-5)
        log_probs = torch.log(probs)

        # calculate confidence (average log probability)
        # SHAPE: (batch_size, seq_len)
        log_probs = torch.gather(log_probs.view(-1, log_probs.size(-1)),
                                 dim=1, index=tags.view(-1, 1)).view(*tags.size())
        log_probs *= mask.float()
        # SHAPE: (batch_size)
        avg_log_probs = log_probs.sum(-1) / (mask.sum(-1).float() + 1e-13)
        output_dict['scores'] = avg_log_probs

        # hinge loss
        if labels is not None:
            half_alp = avg_log_probs / 2
            loss = margin_ranking_loss(half_alp, -half_alp, labels.float(), margin=1.0, reduction='none')
            if weights is not None:
                loss *= weights
            loss = loss.mean()
            output_dict['loss'] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}
