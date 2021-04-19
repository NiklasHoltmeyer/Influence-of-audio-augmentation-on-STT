from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import tensorflow as tf

class Callbacks:
    def __init__(self, batch_size, **kwargs):
        early_stopping_min = kwargs.get("early_stopping_min", 1e-3)
        early_stopping_patience = kwargs.get("early_stopping_patience", 5)
        early_stopping_monitor = kwargs.get("early_stopping_monitor", "val_loss")

        self.earlyStopping = EarlyStopping(
            monitor=early_stopping_monitor,  # loss
            min_delta=early_stopping_min,
            patience=early_stopping_patience,
            restore_best_weights=True)

        # self.reduceLRonPlateau = ReduceLROnPlateau(monitor='val_loss', patience=early_stopping_patience,
        #                                          cooldown=0)

        self.model_checkpoint = lambda checkpoint_path: ModelCheckpoint(
            filepath=checkpoint_path,  # + "/cp-{epoch:04d}.ckpt",
            save_weights_only=True,
            verbose=1,
            save_best_only=True)  # every poch # batch_size*5 = every 5th epoch)

        self.csv_logger = lambda filePath: CSVLogger(filePath, separator=';', append=True)

    def make(self, base_path, model_name):
        checkpoint_path = Callbacks.create_logging_path(str(Path(base_path, "cp").resolve()), model_name)
        csv_path = str(Path(base_path, model_name + ".csv").resolve())
        return [self.earlyStopping, self.model_checkpoint(checkpoint_path), self.csv_logger(csv_path)]

    @staticmethod
    def create_logging_path(model_folder_path, model_dir_name):
        path = Path(model_folder_path, model_dir_name)
        path.mkdir(parents=True, exist_ok=True)

        return str(path.resolve())

    @staticmethod
    def load_model_from_cp(model, checkpoint_path):
        latest_cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
        return model.load(latest_cp)



