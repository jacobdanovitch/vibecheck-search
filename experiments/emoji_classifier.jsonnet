local transformer_model = "distilbert-base-uncased"; //
local transformer_dim = 768;

{
  "dataset_reader": {
    "type": "twitter",
    "sample": 20000,
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
  //"train_data_path": "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/sample.csv",
  "train_data_path": "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/train_multilabel.csv",
  // "validation_data_path": "https://storage.googleapis.com/jacobdanovitch/twitter-emojis/dev_multilabel.csv",
  "model": {
    "type": "bertmoji",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512,
          //"requires_grad": false
        }
      }
    },
    "encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
       "cls_is_last_token": false
    },
    "classifier_feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 3,
      "hidden_dims": [300, 150, 50],
      "activations": ["relu", "relu", "linear"],
      "dropout": [0.5, 0.25, 0.0]
    }
  },
  /*
  "iterator": 
  */
  "data_loader": {
    "type": "default", // "basic",
    "batch_size": 128
  },
  "trainer": {
    "num_epochs": 15,
    "patience": 3,
    "cuda_device": 0,
    // "grad_clipping": 5.0,
    "validation_metric": "+multilabel-f1",// "-loss",
    "optimizer": {
      "type": "huggingface_adamw", 
      "lr": 1e-4, //1e-8,
      "weight_decay": 0.05, //8e-1
      /*
      "parameter_groups": [
          [[], {},],
          [[".*transformer.*"], {"lr": 0.1} ],
      ],
      */
      "parameter_groups": [
        [["classifier.*[0-2].*"], {}],
        // [["classifier.*[1].*"], {}],
        // [["classifier.*[0].*"], {}],
        [[".*transformer\\.layer\\.[45].*"], {}],
        [[".*transformer\\.layer\\.[23].*"], {}],
        [[".*transformer\\.layer\\.[01].*"], {}],
        [[".*word_embeddings.*"], {}],
        [[".*position_embeddings.*"], {},],
        [[".*embeddings\\.LayerNorm.*"], {}],
      ],
    },
    ///*
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "gradual_unfreezing": true,
      "discriminative_fine_tuning": true,
      "cut_frac": 0.25 //0.06
    },
    //*/
    "summary_interval": 25,
    "histogram_interval": 25,
    "num_gradient_accumulation_steps": 10
  }
}
