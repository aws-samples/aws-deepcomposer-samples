# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import torch
from IPython import display
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import seaborn as sns

def save_checkpoint(
        model,
        train_step,
        best_val_loss,
        save_path,
        name="checkpoint.pt",
):
    checkpoint = {
        "model": model.state_dict(), 
        "train_step": train_step,
        "best_val_loss": best_val_loss,
    }

    print(f"Saving checkpoint to {save_path}")
    torch.save(checkpoint, os.path.join(save_path,name))

def plot_losses(train_losses, val_losses, test_losses):
    sns.set()
    display.clear_output(wait=True)
    fig = plt.figure(figsize=(15,5))
    
    def plot_lines(loss_dic, color):
        iters = list(loss_dic.keys())
        vals = [loss_dic[i] for i in iters]
        return plt.plot(iters, vals, color)
    
    line1, = plot_lines(train_losses, 'r')
    line2, = plot_lines(val_losses, 'k')
    line3, = plot_lines(test_losses, 'b')
    
    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.legend((line1, line2, line3), ('train-loss', 'val-loss', 'test-loss'))
    
    # return display.display(fig)
    

