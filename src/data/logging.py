from keras.callbacks import TensorBoard


def init_tensorboard_for_logging(tensorboard_log_dir):
    return TensorBoard(log_dir=tensorboard_log_dir)
