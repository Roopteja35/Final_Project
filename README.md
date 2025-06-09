# Final_Project
This repository consists of three projects.
  1. Comparison of CNN Architectures on Different Datasets
  2. Sequence-to-Sequence Modeling with Attention Mechanism
  3. Multifunctional NLP and Image Generation Tool using Hugging Face Models

PROJECT-1
Comparison of CNN Architectures on Different Datasets

Problem Statement
The goal of this project is to compare the performance of different CNN architectures on various datasets. Specifically, we will evaluate LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, and SENet on MNIST, FMNIST, and CIFAR-10 datasets. The comparison will be based on metrics such as loss curves, accuracy, precision, recall, and F1-score.

Data Set
The datasets used in this project are:
- MNIST: Handwritten digits dataset consisting of 60,000 training images and 10,000 testing images. Each image is 28x28 pixels in grayscale.
- FMNIST: Fashion MNIST dataset consisting of 60,000 training images and 10,000 testing images of fashion products. Each image is 28x28 pixels in grayscale.
- CIFAR-10: Dataset consisting of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 testing images.
Data Set Explanation
The datasets are chosen to cover a variety of image classification tasks:
- MNIST and FMNIST provide simpler tasks with grayscale images, allowing for the evaluation of basic image recognition capabilities.
- CIFAR-10 offers a more complex task with color images, testing the models’ abilities to handle more detailed and varied data.

Approach
1. Load and preprocess the datasets (MNIST, FMNIST, CIFAR-10).
2. Implement the following CNN architectures: LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, and SENet.
3. Train each model on each dataset, recording the loss and accuracy metrics.
4. Evaluate the performance of each model on the test sets using accuracy, precision, recall, and F1-score.
5. Plot the loss curves and other performance metrics for comparison.
6. Analyze the results to understand the impact of different architectures and datasets on model performance.

Results
The expected outcomes of this project include:
- Comparative loss curves for each model on each dataset
- Accuracy, precision, recall, and F1-score for each model on each dataset
- Analysis of the results to determine the strengths and weaknesses of each architecture on different datasets

PROJECT-2
Sequence-to-Sequence Modeling with Attention Mechanism

Problem Statement
The goal of this project is to implement and evaluate sequence-to-sequence (seq2seq) models with attention mechanism. We will train the models on a synthetic dataset where the target sequence is the reverse of the source sequence. The project aims to demonstrate the effectiveness of the attention mechanism in improving seq2seq model performance.

Data Set
The dataset used in this project is a synthetic dataset generated for the purpose of this project. Each source sequence is a random sequence of integers, and each target sequence is the reverse of the source sequence.

Data Set Explanation
The synthetic dataset is chosen to provide a clear and simple example of the sequence-to-sequence modeling task. By reversing the source sequence to obtain the target sequence, we can easily evaluate the model's ability to learn the seq2seq mapping.

Approach
1. Generate a synthetic dataset where each source sequence is a random sequence of integers, and each target sequence is the reverse of the source sequence.
2. Implement the sequence-to-sequence model with attention mechanism in PyTorch.
3. Train the model on the synthetic dataset.
4. Evaluate the model performance using metrics such as loss and accuracy.
5. Plot the loss curves and other performance metrics for analysis.
6. 
Results
The expected outcomes of this project include:
- Loss curves for the seq2seq model with attention mechanism during training.
- Accuracy of the model in predicting the target sequences from the source sequences.
- Analysis of the effectiveness of the attention mechanism in improving seq2seq model performance.

PROJECT-3
Multifunctional NLP and Image Generation Tool using Hugging Face Models

Problem Statement
The goal of this project is to create a multifunctional tool that allows users to select and utilize different pretrained models from Hugging Face for various tasks. The tool will support text summarization, next word prediction, story prediction, chatbot, sentiment analysis, question answering, and image generation. The front end will provide a user-friendly interface to select the task and input the required text or image for processing.

Data Set
The project will utilize pretrained models from Hugging Face, which have been trained on extensive datasets. No additional dataset is required as the models come with pre-trained weights for the tasks.

Data Set Explanation
The pretrained models from Hugging Face have been trained on diverse and extensive datasets, providing robust performance for various NLP tasks. These models include GPT-3, BERT, T5, GPT-2, and others, each specialized for different tasks such as text summarization, next word prediction, story prediction, chatbot, sentiment analysis, question answering, and image generation.

Approach
1. Set up the environment and install necessary libraries, including Hugging Face Transformers.
2. Implement a user-friendly front end for task selection and input.
3. Load and integrate pretrained models from Hugging Face for the following tasks:
   - Text Summarization
   - Next Word Prediction
   - Story Prediction
   - Chatbot
   - Sentiment Analysis
   - Question Answering
   - Image Generation
4. Implement the backend logic to process user inputs and generate outputs using the selected models.
5. Evaluate the model performance using appropriate metrics for each task.
6. Test the application with various inputs and refine the user interface and backend logic.

Results
The expected outcomes of this project include:
- A functional application that allows users to select and utilize different NLP and image generation models.
- Evaluation of model performance for each task, with metrics such as accuracy, precision, recall, F1-score, and user satisfaction.
- Analysis of the effectiveness of integrating multiple models into a single application.
