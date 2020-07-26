# Welcome
## What is AWS DeepComposer? 
 AWS DeepComposer provides a creative, hands-on experience for learning generative AI and machine learning. 
 With generative AI, one of the biggest recent advancements in artificial intelligence, you can create a new dataset
 based on a training dataset. With AWS DeepComposer, you can experiment with different generative AI architectures
 and models in a musical setting by creating and transforming musical inputs and accompaniments.

Regardless of your experience with machine learning (ML) or music, you can use AWS DeepComposer to develop a working
knowledge of generative AI. AWS DeepComposer includes learning capsules, sample code, and training data to help you
understand and use generative AI models. 

**Note** 
> To use the AWS DeepComposer console and other AWS services, you need an AWS account. If you don't have an account, 
>go to aws.amazon.com and choose **Create an AWS Account**. For detailed instructions, see [Create and Activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/).

>As a best practice, you should also create an AWS Identity and Access Management (AWS IAM) user with administrator
>permissions and use that for all work that does not require root credentials. 
>Create a password for console access, and access keys to use command line tools. For more information, see [Creating Your First IAM Admin User and Group](https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html) in the *AWS IAM User Guide*.

## Prequisites 

In order to best understand this notebook we recommend that you go through the following learning capsules 
1. [Introduction to autoregressive convolutional neural networks](https://console.aws.amazon.com/deepcomposer/home?region=us-east-1#learningCapsules/autoregressive)
2. [A deep dive into training an autoregressive convolutional neural network model AR-CNN model](https://console.aws.amazon.com/deepcomposer/home?region=us-east-1#learningCapsules) 

Additionally, if you are unfamiliar with Amazon SageMaker check out the [Getting started with Amazon SageMaker notebook
instances and SDKs](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html)  

###  Getting Started 
To begin creating your custom genre-based **AR-CNN** model you need to launch an Amazon SageMaker instance.

**Note**
>Jupyter notebooks are open-source web applications that you can use to create and share documents that contain live code, equations, visualizations and narrative text.
>The AWS DeepComposer jupyter notebook in this repo contains code that demonstrates how to create machine learning solutions with Amazon SageMaker and AWS DeeComposer.

**Creating an Amazon SageMaker Training Instance**

1. Open the [Amazon SageMaker console](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/dashboard).

2. In the navigation pane, choose **Notebook instances**

3. On the **Notebook instances** page, choose **Create notebook instance**.

4. On the **Create notebook instance** page, enter your notebook name in **Notebook instance name**, and then choose a default `ml.p3.2xlarge` instance.

5. Choose your **IAM Role** to set up the correct permissions and encryption. If you have an existing Amazon 
SageMaker IAM role, select that IAM role from the list. If you are new to **Amazon SageMaker**, create an IAM role by choosing **Create a new role**.

6. On the **Create an IAM role** page, choose **Any S3 bucket** to give your new IAM role access to your S3 bucket.

7. Choose **Create Role**. 

8. On the **Create notebook instance** page, choose the **IAM Role** dropdown menu, and then choose your newly created IAM role from the list.

9. Next, open the **Github repositories** panel. Then, under the **Default repository** dropdown choose **Clone a public Git repository to this notebook instance only**.

10. This git repository[project-title](https://github.com/froggie901/deep_lens_project) contains the Jupyter notebook 
required for this custom project. Copy this link and paste it into the provided field.

11. Then, choose **Create notebook instance**.

12. On the **Notebook instances** page, choose **Open Jupyter** to launch your newly created Jupyter notebook.

13. On the **jupyter screen**, choose **New* and then choose the `conda_tensorflow_p36` virtual environment which will launch
your new Jupyter notebook.

## Legal

```
Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

