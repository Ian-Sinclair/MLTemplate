from src.main.classes.modelArchitectures import BaseModel
from src.main.classes.modelArchitectures.CNN import CNNMetrics
from src.main.classes.projectUtils import DirectoryUtil
from src.main.classes.modelArchitectures.CNN.CNNVisualizations import CNNVisualizations

import pickle
from keras import Model, layers, Input
from keras import optimizers
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

from typing_extensions import Self
from typing import Literal, Callable


class CNN(BaseModel):
    MODEL_NAME = "CNN"
    CLASS_FILENAME = "WGAN_GP.pkl"
    TRAINING_METRICS_FILENAME = "training_metric.csv"

    def __init__(self, 
                 input_size : tuple,
                 filter_sizes : list[int], 
                 kernel_sizes : list[int], 
                 strides : list[int], 
                 output_dim : int,
                 activation_functions : list[Callable | str | Literal["linear", "relu", "sigmoid", "tanh", "softmax"]] = ['relu'],
                 output_activation_function : Callable | str | Literal["linear", "relu", "sigmoid", "tanh", "softmax"] = "softmax",
                 use_batch_norm : bool = True, 
                 use_drop_out : bool = True,
                 drop_out_rate : float = 0.25,
                 name : str = ""
                 ):

        # Initialize metric classes
        self.loss_metrics = CNNMetrics.loss_metrics()
        self.time_metrics = CNNMetrics.time_metrics()
        
        self.input_size = input_size
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.output_dim = output_dim
        self.activation_functions = activation_functions
        self.output_activation_function = output_activation_function
        self.use_batch_norm = use_batch_norm
        self.use_drop_out = use_drop_out
        self.drop_out_rate = drop_out_rate
        self.name = name

        self.model_depth = min(len(lst) for lst in [self.filter_sizes, self.kernel_sizes, self.strides])

        self.model : Model = self._build_model()

        super().__init__(self.model)

        
    def _build_model(self) -> Model :
        """
        Build the CNN model.

        Returns:
            Model: The CNN model.
        """
        
        # -------------------------
        # ----Input Validation ----
        # -------------------------
        if not (len(self.filter_sizes) == len(self.kernel_sizes) == len(self.strides)) :
            print(f"WARNING: Layer Diminsionality Is Un-Even - number of each layer = filters - [{len(self.filter_sizes)}, kernels - [{len(self.kernel_sizes)}], strides - [{len(self.strides)}]]")
            print(f"WARNING: Reconfiguring Layers")
            self.filter_sizes = self.filter_sizes[self.model_depth:]
            self.kernel_sizes = self.kernel_sizes[self.model_depth:]
            self.strides = self.strides[self.model_depth:]
            print(f"filter_sizes = [{self.filter_sizes}, kernel_sizes = [{self.kernel_sizes}], strides = [{self.strides}]]")
        
        if len(self.activation_functions) != self.model_depth :
            print(f"NOTICE: Extending Activation Function Array")
            self.activation_functions = self.activation_functions + [self.activation_functions[-1]]*(self.model_depth-len(self.activation_functions))

            print(f"Activation Function Length: [{len(self.activation_functions)}]")

        # --------------------------
        # --- Model Construction ---
        # --------------------------
        
        self.input_layer : layers.Input = Input(shape=self.input_size, name="Input")

        x : layers.Input | layers.Layer = self.input_layer
        for filter, kernel, stride, activation_function, index in zip(self.filter_sizes, self.kernel_sizes, self.strides, self.activation_functions, range(0,len(self.filter_sizes))):
            conv_layer : layers.Layer = layers.Conv2D(filters=filter,
                                      kernel_size=kernel,
                                      strides=stride, 
                                      padding="same", 
                                      name=f'{self.name}Conv2D_{index}')
            x = conv_layer(x)
            if self.use_batch_norm :
                x = layers.BatchNormalization()(x)
            
            x = layers.Activation(activation_function)(x)

            if self.use_drop_out :
                x = layers.Dropout(rate=self.drop_out_rate)(x)
            
        x = layers.Flatten()(x)

        self.output_layer : layers.Layer = layers.Dense(
            self.output_dim, 
            activation=self.output_activation_function, 
            name = "output_layer")(x)
            
        model : Model = Model(self.input_layer, outputs=self.output_layer, name=self.name)

        return model

    
    def train(self, 
              x_train : np.ndarray, 
              y_train : np.ndarray, 
              batch_size : int, 
              num_epochs : int,
              optimizer : optimizers.Optimizer = optimizers.legacy.Adam,
              learning_rate : float = 0.0005,
              save_folder : str = None
              ) -> Self :

        total_runtime_start_time = time.time()
        
        loss_plot_handler = CNNVisualizations.Line_Plot_Handler(
            x_label="Step",
            y_labels=self.loss_metrics.toJSON().keys()
            )
        
        num_batches = x_train.shape[0]//batch_size
        
        #  Initialize Optimizer
        optimizer = optimizer(learning_rate)

        print(f"Initializing Training Cycle \n Epochs {num_epochs}, steps per epoch {x_train.shape[0]//batch_size}")

        for epoch in range(num_epochs):
            for step in range(num_batches):
                overhead_start_time = time.time()
                #  Create batch
                start_idx = batch_size*step
                end_idx = batch_size*(step+1)
                X_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                training_start_time = time.time()
                self.loss_metrics.mse_loss = self._train_batch(X_batch = X_batch, y_batch = y_batch, optimizer=optimizer)
                self.time_metrics.training_step_time = time.time() - training_start_time

                loss_plot_handler\
                    .update_plot(x_data=(epoch*num_batches)+step, y_data=self.loss_metrics.reduce_tensors().toJSON())\
                    .save_plot(save_path=os.path.join(save_folder,"viz/"))
                
                self.time_metrics.overhead_step_time = (time.time() - overhead_start_time) - self.time_metrics.training_step_time
                self.time_metrics.total_runtime = time.time() - total_runtime_start_time

                # save metrics
                if save_folder is not None :
                    self._save_metrics(epoch=epoch, step=step, save_folder=save_folder, initialize_file=(epoch==0 and step==0))

                # print training summary
                self._print_training_summary(epoch, num_epochs, step, num_batches)
        
        return self
    
    @tf.function()
    def _train_batch(
            self, 
            X_batch : np.ndarray, 
            y_batch : np.ndarray, 
            optimizer : optimizers.Optimizer) :
        with tf.GradientTape() as tape:
            y_prob = self.model(X_batch)
            mse_loss = tf.keras.losses.categorical_crossentropy(y_batch, y_prob)

            grads = tape.gradient(mse_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return mse_loss
    
    def _save_metrics(self, epoch, step, save_folder : str, initialize_file = False):
        metrics_path = os.path.join(save_folder, "metrics/")
        if not DirectoryUtil.isValidDirectory(metrics_path):
            print(f"Directory Not Found at [{metrics_path}]")
            try : 
                DirectoryUtil.promptToCreateDirectory(metrics_path) # Throws value error
            except ValueError as e:
                print(e)
                return self
        
        if DirectoryUtil.isProtectedFile(metrics_path,CNN.TRAINING_METRICS_FILENAME) :
            print(f"ABORTING: File [{os.path.join(metrics_path, CNN.TRAINING_METRICS_FILENAME)}] is protected - cannot overwrite file")
            return self
        
        metrics = [{"Epoch" : epoch, "Step" : step, **self.loss_metrics.reduce_tensors().toJSON(), **self.time_metrics.toJSON()}]

        df = pd.DataFrame(metrics)

        csv_file = os.path.join(metrics_path, CNN.TRAINING_METRICS_FILENAME)
        # Save DataFrame to CSV with headers
        if initialize_file :
            df.to_csv(csv_file, index=False)
        else : 
            df.to_csv(csv_file, mode='a', header=False, index=False)
    
    def _print_training_summary(self, epoch, epochs, step, steps):
        loss_JSON = self.loss_metrics.reduce_tensors().toJSON()
        loss_report = "".join(f"""
        %   {str(key)}          {value}
        """ for key, value in loss_JSON.items())

        time_JSON = self.time_metrics.toJSON()
        time_report = "".join(f"""
        %   {str(key)}          {value}
        """ for key, value in time_JSON.items())

        summary : str = f"""
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%% Training Summary %%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %   Epoch:  {epoch + 1}/{epochs}
        %   Step:   {step+1}/{steps}
        %   
        %%%%   Critic Losses   %%%%
        {loss_report}
        %
        %%%%   Time     %%%%
        {time_report}
        """

        if epoch == 0 and step == 0 : print("\n" * (summary.count('\n') + 1), end='', flush=True)
        print("\033[F" * (summary.count('\n') + 1) + summary, flush=True)
        return True
    
    def save_architecture_diagram(self, path: str, file_name: str) -> Self:
        """
        Saves the architecture diagrams of the complete model.

        Args:
            path (str): Path to the directory to save the diagrams.
            file_name (str): Prefix for the file names of the diagrams.

        Returns:
            Self: The instance of the class.
        """
        return super().save_architecture_diagram(path=path, file_name=file_name+ "_" + CNN.MODEL_NAME + ".png")
    
    def save_model(self, path: str) -> Self:
        """
        Saves the model and its components to the specified path.

        Args:
            path (str): Path to the directory to save the model.

        Returns:
            Self: The instance of the class.
        """
        if not DirectoryUtil.isValidDirectory(path):
            print(f"Directory Not Found at [{path}]")
            try : 
                DirectoryUtil.promptToCreateDirectory(path) # Throws value error
            except ValueError as e:
                print(e)
                return self
        
        if DirectoryUtil.isProtectedFile(path,CNN.CLASS_FILENAME) :
            print(f"ABORTING: File [{os.path.join(path, CNN.CLASS_FILENAME)}] is protected - cannot overwrite file")
            return self

        # Serialize and save the instance data as JSON
        with open(os.path.join(path, CNN.CLASS_FILENAME), 'wb') as file:
            pickle.dump(self, file)

        return super().save_model(path=os.path.join(path, CNN.MODEL_NAME + "/"))
    
    @staticmethod
    def load_model(path : str) -> Self:
        """
        Loads the model from the specified path.

        Args:
            path (str): Path to the directory containing the saved model.

        Returns:
            Self: The instance of the class.
        """
        file_path = os.path.join(path, CNN.CLASS_FILENAME)
        with open(file_path, 'rb') as file:
             instance = pickle.load(file)
        
        self = instance
        return self
    
    def __str__(self) -> str:
        """
        Returns a string representation of the WGAN_GP object.

        Returns:
            str: String representation of the object.
        """
        stringSummary = []
        stringSummary.append(super().__str__())
        return "\n".join(stringSummary)

        
