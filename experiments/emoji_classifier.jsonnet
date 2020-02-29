local VOCAB = {
    "type": "from_files",
    "directory": std.extVar("VOCAB_DIR")
};

local transformer_model = "albert-base-v2"; // "distilbert-base-uncased";

{
  "dataset_reader": {
    "type": "twitter",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer", // _mismatched",
        "model_name": transformer_model
      }
    }
  },
  "train_data_path": "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/sample.csv", // "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/train_multilabel.csv",
  // "validation_data_path": "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/dev_multilabel.csv",
  "vocabulary": VOCAB,
  "model": {
    "type": "bertmoji",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 768,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 200,
      "num_layers": 2,
      "hidden_dims": [200, 50],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },
  /*
  "iterator": 
  */
  "data_loader": {
    "type": "default", // "basic",
    "batch_size": 2
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "-loss",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
