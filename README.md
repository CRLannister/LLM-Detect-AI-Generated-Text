# LLM---Detect-AI-Generated-Text Project Documentation
This project fulfills the requirements for the final project of DSCI-6612 Introduction to AI, instructed by Professor Vahid Behzadan, as part of the Master's in Data Science program at the University of New Haven. The dataset used in this project was obtained from Kaggle.

## Project Overview

In recent years, large language models (LLMs) have advanced to the point where they can generate text that closely resembles human writing. This project aims to promote open research and transparency in the field of AI detection techniques. Specifically, this project deals with developing a machine learning model capable of accurately distinguishing between essays written by students and those generated by LLMs.

## Problem Statement

The primary goal of this project is to build a model that can identify whether an essay was authored by a middle or high school student or if it was generated by a large language model. The increasing sophistication of LLMs has raised concerns about their potential impact on education, particularly in terms of skill development and the prevention of plagiarism.

## Dataset

The dataset consists of essays written by students and essays generated by LLMs. The essays in our dataset cover 2 topics and are of moderate length. The dataset is sourced frm kaggle[!https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview].

## Project Components

The project is structured into the following components:

1. **Data Loading and Exploration:** The initial phase involves loading the dataset and exploring its structure. This includes understanding the features, distributions, and characteristics of the provided essays.

2. **Text Feature Engineering:** Textual features are extracted from the essays to aid in the training of the detection model. Features such as spelling errors, punctuation errors, and collocations are considered.

3. **Training the Model:** A Neural network model is trained on the engineered features to distinguish between student-written essays and those generated by LLMs. The model is evaluated using cross-validation techniques.

4. **Evaluation Metrics:** Various metrics, including accuracy, precision, recall, and F1 score, are used to evaluate the performance of the detection model.

5. **ROC AUC Analysis:** The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are utilized to assess the model's ability to discriminate between student and LLM-generated essays.

6. **Testing and Submission:** The trained model is applied to a separate set of essays to predict whether they were written by students or LLMs. The results are then submitted for evaluation.

## Usage Instructions

### Prerequisites

You can run this Notebook on Google Colab or in your local machine

Make sure you have the following dependencies installed:

- Python (>=3.6)
- Required Python libraries (install using `pip install -r requirements.txt`)

### Steps to Replicate Results

1. **Clone the Repository:**
   ```bash
   git clone "https://github.com/CRLannister/LLM---Detect-AI-Generated-Text.git"
   cd "LLM---Detect-AI-Generated-Text"
    ```
2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Explore the Notebook:**

    Explore the Jupyter notebook provided in the repository to understand each step of the project.

4.  **Train the Model:**

    Run the notebook for training the Neural Network model. This notebook includes data loading, feature engineering, model training, and evaluation steps.

5.  **Evaluate Model Performance:**

    Review the metrics and visualizations in the notebook to understand how well the model performs on the validation set.

6.  **Test the Model:**

    Use the trained model to predict whether a given essay is student-written or LLM-generated.

## Conclusion
This documentation provides an overview of the LLM Detection Project, its objectives, and detailed instructions for replicating the results. Feel free to explore the provided notebook for a more in-depth understanding of each project phase.

For any questions or clarifications, please feel free to contact me @linkedin[!https://www.linkedin.com/in/agarwalashishsinghal/]

Happy exploring and detecting LLMs! 🚀
