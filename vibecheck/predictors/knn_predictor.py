from typing import List, Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

import pandas as pd
import numpy as np
from annoy import AnnoyIndex

import os
from urllib.request import urlretrieve
from tqdm.auto import tqdm

@Predictor.register('knn_classifier')
class KNNPredictor(Predictor):
    def __init__(self, 
                 model: Model, 
                 dataset_reader: DatasetReader,
                 vocab_path: str = 'resources/vocab',
                 df_path: str = 'https://storage.googleapis.com/jacobdanovitch/spotify_lyrics/spotify_with_genius.csv',
                 annoy_index_path: str = 'https://storage.googleapis.com/jacobdanovitch/spotify_lyrics/index.tree'
                ) -> None:
        super().__init__(model.eval(), dataset_reader)
        
        self.vocab = Vocabulary().from_files(vocab_path)
        self.df = pd.read_csv(df_path).set_index("track_id")
        
        self.index = None
        if annoy_index_path:
            self.build_index(annoy_index_path)
    
    def build_index(self, path: str, tracks: List[Tuple[str, np.array]] =None):
        features = self._model.classifier_feedforward.get_output_dim()
        if tracks is None:
            if not os.path.exists(path):
                path = urlretrieve(path)[0]
            self.index = AnnoyIndex(features, metric='angular')
            self.index.load(path)
            return
        
        index = AnnoyIndex(features, metric='angular')
        for track, vector in tqdm(tracks):
            i = self.vocab.get_token_to_index_vocabulary("labels")[track]
            index.add_item(i, vector)
        
        index.build(-1)
        index.save(path)
        
        self.index = index
    
    def neighbors_to_tracks(self, nns):
        tracks = [self.vocab.get_token_from_index(i, "labels") for i in nns]
        return self.df.loc[tracks].reset_index(drop=True).to_dict(orient='records')
    
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        n = inputs.pop('n', 10)
        if 'track_id' in inputs:
            if self.index is None:
                raise AttributeError("Please build an index before searching by track.")
            idx = self.vocab.get_token_to_index_vocabulary("labels")[inputs['track_id']]
            nns = self.index.get_nns_by_item(idx, n+1)[1:]
            #scores = self.index.get_item_vector(idx) 
            tracks = self.neighbors_to_tracks(nns)
            return tracks
            #return {'tracks': tracks, 'scores': scores}
            
            
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        output_dict['inputs'] = inputs
        if self.index:
            logits = output_dict.get('logits')
            nns = self.index.get_nns_by_vector(logits, n)
            return self.neighbors_to_tracks(nns)
            #output_dict['tracks'] = self.neighbors_to_tracks(nns)
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(text=json_dict['query'])
