# The MIT-Zero License

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import matplotlib.pyplot as plt
from IPython.display import clear_output
import keras
import numpy as np


class GenerateTrainingPlots(keras.callbacks.Callback):

    # Generates Training Vs Validation live plots
    def on_train_begin(self, logs={}):

        # Create training and validation loss lists
        self.training_loss = []
        self.validation_loss = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.training_loss.append(logs.get('loss'))
        self.validation_loss.append(logs.get('val_loss'))

        # Start generating plot after Epoch 2
        if len(self.training_loss) > 1:
            clear_output(wait=True)
            N = np.arange(0, len(self.training_loss))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.training_loss, label="training_loss")
            plt.plot(N, self.validation_loss, label="validation_loss")
            plt.title(
                "Training Vs Validation Loss After Epoch {}".format(epoch + 1))
            plt.xlabel("Epoch Number")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
