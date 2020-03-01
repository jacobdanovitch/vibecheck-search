from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

import pandas as pd
from annoy import AnnoyIndex

import os

@Predictor.register('knn_classifier')
class KNNPredictor(Predictor):
    def __init__(self, 
                 model: Model, 
                 dataset_reader: DatasetReader,
                 vocab_path: str = 'resources/vocab',
                 annoy_index_path: str = 'resources/index.tree',
                 df_path: str = 'https://storage.googleapis.com/jacobdanovitch/spotify_lyrics/spotify_with_genius.csv'
                ) -> None:
        super().__init__(model, dataset_reader)
        
        self.vocab = Vocabulary().from_files(vocab_path)
        self.df = pd.read_csv(df_path).set_index("track_id")
        
        self.index = AnnoyIndex(self._model.classifier_feedforward.get_output_dim(), metric='angular')
        self.index.load(annoy_index_path)
    
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        n = inputs.get('n', 10)
        if 'track_id' in inputs:
            idx = self.vocab.get_token_to_index_vocabulary("labels")[inputs['track_id']]
            nns = self.index.get_nns_by_item(idx, n+1)[1:]
            tracks = [self.vocab.get_token_from_index(i, "labels") for i in nns]
            return self.df.loc[tracks].reset_index(drop=True).to_dict(orient='records')
            
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        logits = output_dict.pop('logits')
        nns = self.index.get_nns_by_vector(logits, n)
        tracks = [self.vocab.get_token_from_index(i, "labels") for i in nns]
        return self.df.loc[tracks].reset_index(drop=True).to_dict(orient='records')

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(text=json_dict['query'])
