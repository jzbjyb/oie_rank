// Configuration for RnnOIE
{
  "dataset_reader": {
    "type": "srl",
    "token_indexers": {
      "elmo": {"type": "elmo_characters"}
    }
  },
  "train_data_path": "data/train",
  "validation_data_path": "data/dev",
  "model": {
    "type": "srl",
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
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
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
    "num_epochs": 200,
    "grad_clipping": 1.0,
    "patience": 10,
    "num_serialized_models_to_keep": 10,
    "validation_metric": "+f1-measure-overall",
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
