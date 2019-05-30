// Configuration for the iterative rank-aware training
{
  "dataset_reader": {
    "type": "rerank",
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    }
  },
  "train_data_path": "ITER_DATA_ROOT/oie2016.train.iter.conll",
  "validation_data_path": "ITER_DATA_ROOT/oie2016.dev.iter.conll",
  "model": {
    "type": "reranker",
    "text_field_embedder": {
      "elmo": {
        "type": "elmo_token_embedder",
        "options_file": "pretrain/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
        "weight_file": "pretrain/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
        "do_layer_norm": false,
        "dropout": 0.1
      }
    },
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 1124,
      "hidden_size": 64,
      "num_layers": 4,
      "recurrent_dropout_probability": 0.1,
      "use_input_projection_bias": false
    },
    "binary_feature_dim": 100,
    "initializer": [
      [
        ".*",  // initialize all parameters from an open IE model
        {
          "type": "pretrained",
          "weights_file_path": "ITER_MODEL_ROOT/weights.th"
        }
      ]
    ],
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },
  "trainer": {
    "num_epochs": 1,
    "grad_clipping": 1.0,
    "patience": 1,
    "num_serialized_models_to_keep": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  },
  "vocabulary": {
    "directory_path": "pretrain/vocabulary"
  }
}
