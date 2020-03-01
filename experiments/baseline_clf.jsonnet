{
  "dataset_reader": {
    "type": "twitter",
    "sample": 250000,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      },
    }

  },
  //"train_data_path": "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/sample.csv",
  "train_data_path": "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/train_multilabel.csv",
  "validation_data_path": "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/dev_multilabel.csv",
  "model": {
    "type": "bertmoji",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
          "embedding_dim": 300,
          "trainable": true
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "embedding_dim": 200
            },
            "encoder": {
                "type": "lstm",
                "input_size": 200,
                "hidden_size": 50,
                "bidirectional": true
            }
        }
      }
    },
    "encoder": {
       "type": "lstm",
       "input_size": 400,
       "hidden_size": 400,
       "bidirectional": true
    },
    "classifier_feedforward": {
      "input_dim": 800,
      "num_layers": 3,
      "hidden_dims": [400, 200, 50],
      "activations": ["relu", "relu", "linear"],
      "dropout": [0.2, 0.1, 0.0]
    }
  },
  "data_loader": {
    "type": "default", // "basic",
    "batch_size": 1024
  },
  "trainer": {
    "num_epochs": 50,
    //"patience": 5,
    "cuda_device": 0,
    "validation_metric": "+multilabel-f1",
    "optimizer": {
      "type": "adam",
      "lr": 0.01,
    },
    ///*
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    //*/
    "summary_interval": 25,
    "histogram_interval": 25
  }
}