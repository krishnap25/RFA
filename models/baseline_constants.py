MAIN_PARAMS = {  # (tot_num_rounds, eval_every_num_rounds, clients_per_round)
    'sent140': (10, 2, 2),
    'femnist': (400, 20, 2),
    'shakespeare': (20, 1, 2)
}

MODEL_PARAMS = {
    'sent140.bag_dnn': (0.0003, 2, round(1e9)),  # lr, num_classes, max_batch_size
    'sent140.stacked_lstm': (0.0003, 25, 2, 100, 16384),  # lr, seq_len, num_classes, num_hidden, max-batch-size
    # TODO: find max batch size of sent140.stacked_lstm'
    'sent140.bag_log_reg': (0.0003, 2, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.cnn': (5e-2, 62, 16384),  # lr, num_classes, max_batch_size
    'femnist.log_reg': (2e-2, 62, round(1e9)),  # lr, num_classes, max_batch_size
    # lr, seq_len, num_classes, num_hidden, num_lstm_layers, max_batch_size
    'shakespeare.stacked_lstm': (0.64, 20, 53, 128, 1, 32768)
}

MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount


class OptimLoggingKeys:
    TRAIN_ACCURACY_KEY = 'train_accuracy'
    TRAIN_LOSS_KEY = 'train_loss'
    EVAL_ACCURACY_KEY = 'test_accuracy'
    EVAL_LOSS_KEY = 'test_loss'


TRAINING_KEYS = {OptimLoggingKeys.TRAIN_ACCURACY_KEY,
                 OptimLoggingKeys.TRAIN_LOSS_KEY,
                 OptimLoggingKeys.EVAL_LOSS_KEY}

CORRUPTION_FLIP_KEY = 'flip'
CORRUPTION_OMNISCIENT_KEY = 'omniscient'
CORRUPTION_P_X_KEY = 'p_x'

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
