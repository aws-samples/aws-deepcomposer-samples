# Train a custom GAN model

## Goal

As part of this repo, you will learn to build a custom GAN architecture and train the model using Amazon SageMaker.



## Prerequisites

* Access to Amazon SageMaker

## Cost

Using a ml.c5.4xlarge, the entire exercise take 3-4 hrs to run. Please see the [Amazon SageMaker pricing](https://aws.amazon.com/sagemaker/pricing/) for details. 

## Setup

First we create the Amazon SageMaker notebook instance.

Navigate to Amazon SageMaker using the link: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/dashboard

![notebook-instance](images/notebook-instance.PNG)

Click **Notebook instances** from the left navigation bar

Select **Create notebook instance**

![create-notebook](images/create_notebook.png)

Within the notebook instance creation form, select "c5.4xlarge" for **Notebook instance type**

![notebook-instance-settings](images/notebook_instance_settings.png)

Set the following for **Permissions and encryption**:
* IAM role: Use an existing role or create a new role
* Root access: Enable
* Encryption key: No Custom Encryption

![notebook-instance-settings](images/permissions_and_encryption.png)

Set the following for **Git repositories**:
* Repository: Clone a public Git repository to this notebook instance only
* Git repository URL: https://github.com/aws-samples/aws-deepcomposer-samples

![notebook-instance-settings](images/notebook_git_settings.png)

Click **Open Jupyter**

![open-notebook](images/open_jupyter.png)



Click **gan** folder, then click **GAN.ipynb**

![GAN-notebook](images/gan_notebook.png)

*You will likely be prompted to select kernel. Choose the drop down and select **conda_python3** as the kernel*

![set-kernel](images/set-kernel.PNG)

This notebook contains instructions and code to create a custom GAN model from scratch. Follow the notebook content and run all cells to the end.

![run-notebook](images/run-notebook.PNG)

To run the code cells, choose the code cell you want to run and click **Run**

![kernel-empty](images/kernel-empty.png)

If the kernel has an empty circle, it means it is free and ready to execute the code 

![kernel-busy](images/kernel-busy.png)

If the kernel has a filled circle, it means it is busy. Wait for it to become free before you execute the next line of code.

## Next Steps

**Congratulations on building a custom GAN model from scratch!** 

Now try using your model to create compositions based on your custom MIDI input.



**Important: Remember to stop your Amazon SageMaker instances after you're done to avoid extra charges**



![notebook-stop](images/notebook-stop.png)