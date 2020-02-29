from typing import Dict
import logging

import json
import pandas as pd
import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MultiLabelField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("twitter")
class TwitterDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        data = pd.read_csv(file_path).fillna({'label': 'N/A'}).to_dict(orient='records')
        instances = map(lambda x: self.text_to_instance(**x), data)
        # yield from filter(None, instances)
        for inst in instances:
            if inst:
                yield inst

    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized = self._tokenizer.tokenize(text)
        if not (text and tokenized):
            return None

        fields = {'tokens': TextField(tokenized, self._token_indexers)}
        if label is not None:
            label = label.split(', ')
            fields['label'] = MultiLabelField(label)
        return Instance(fields)
