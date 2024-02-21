from dataclasses import dataclass
import tensorflow as tf

from typing_extensions import Self

@dataclass
class loss_metrics:
    mse_loss : tf.Tensor | float = None

    def reduce_tensors(self) -> Self:
        if isinstance(self.mse_loss, tf.Tensor):
            self.mse_loss = tf.math.reduce_mean(self.mse_loss).numpy()
        return self

    def toJSON(self) :
        return {
            "MSE loss" : self.mse_loss
        }


@dataclass
class time_metrics:
    training_step_time : float = None
    overhead_step_time : float = None
    total_runtime : float = None

    def toJSON(self) :
        return {
            "Training Time Per Step" : self.training_step_time,
            "Overhead Time Per Step" : self.overhead_step_time,
            "Total Runtime" : self.total_runtime
        }