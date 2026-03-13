class Callback:
    def on_epoch_end(self, trainer, epoch: int, logs: dict):
        pass

    def on_validation_batch_end(self, trainer, epoch: int, x, y, prediction):
        pass

    def on_validation_images_end(self, trainer, epoch: int, list_imgs):
        pass
