from typing import Dict, Optional
import logging
from itertools import groupby

import numpy as np
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util

from allennlp.training.metrics import BooleanAccuracy, PearsonCorrelation
from vibecheck.training.metrics.multilabel_f1 import MultiLabelF1Measure

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def rsq_loss(x, y):
    # https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/2
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost

@Model.register("bertmoji")
class BERTMoji(Model):
    """
    This ``Model`` performs text classification for an academic paper.  We assume we're given a
    title and an abstract, and we predict some output label.

    The basic model structure: we'll embed the title and the abstract, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the text to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, 
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        # raise ValueError(self.vocab.get_vocab_size("tokens"))
        # raise ValueError(text_field_embedder.get_output_dim())

        if text_field_embedder.get_output_dim() != encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            encoder.get_input_dim()))

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward

        self.metrics = {
                "multilabel-f1": MultiLabelF1Measure(),
                'accuracy': BooleanAccuracy()
        }
        self.pearson_r = PearsonCorrelation()
        self.loss = nn.MultiLabelSoftMarginLoss() #BCEWithLogitsLoss() 
        
        self._threshold = 0.5

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # print(tokens)
        embedded = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        encoded = self.encoder(embedded, mask)

        logits = self.classifier_feedforward(encoded)
        output_dict = {'logits': torch.sigmoid(logits)}
        
        if label is None: # inference
            decoded = self.decode(output_dict)
            output_dict['decoded'] = decoded
        else:
            loss = self.loss(logits, label.float())
            loss = loss + (1-rsq_loss(logits, label.float()))
            
            self.pearson_r(logits, label.float())
            preds = (logits > self._threshold).long()
            for metric in self.metrics.values():
                metric(preds, label)
            output_dict["loss"] = loss

        return output_dict

    # @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        # class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        # output_dict['class_probabilities'] = class_probabilities
        
        class_probabilities = output_dict['logits']

        predictions = class_probabilities.cpu().data.numpy()
        # argmax_indices = np.argmax(predictions, axis=-1)
        
        scores = ((self.vocab.get_token_from_index(i, namespace="labels"), s) for (i, s) in enumerate(predictions.squeeze()))
        output_dict['scores'] = [sorted(scores, key=lambda x: x[1], reverse=True)]
        
        """
        positive = np.argwhere(predictions > self._threshold)# .squeeze()
        groups = groupby(positive.tolist(), lambda x: x[0])
        label_dict = {key: [self.vocab.get_token_from_index(x[1], namespace="labels") for x in group] for (key, group) 
                      in groups}
        labels = [x[1] for x in sorted(label_dict.items())]
        
        #labels = [self.vocab.get_token_from_index(x, namespace="labels")
        #          for x in argmax_indices]
        output_dict['label'] = [labels]
        """
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        def unpack(m):
            if isinstance(m, tuple):
                return m[-1]
            return m
        metrics = {metric_name: unpack(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
        metrics['pearson_r'] = self.pearson_r.get_metric(reset)
        return metrics
