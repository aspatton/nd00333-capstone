# Udacity ML Engineer with Microsoft Azure - Capstone Project

This is the Capstone project for the Udacity Course - ML Engineer with Microsoft Azure.

This project provides the opportunity to use the knowledge obtained from the course to solve an interesting problem. In this project, two models are created: one using AutoML plus one customized model whose hyperparameters are tuned using HyperDrive. The performance of both the models is compared/contrasted and best performing model is deployed. This project demonstrates the ability to use external datasets, train a model using different AzureML framework tools available as well as the ability to deploy the model as a web service.

Both Hyperdrive and AutoML API are used in this project.

### Problem Statement

In this project we attempt to solve the problem of classifying penguin species for a given input data. We first apply AutoML where multiple models are trained to fit the training data. Secondly, we use a LogicalRegression model while tuning hyperparameters using HyperDrive. Finally, the best model from the two approaches is chosen (in terms of accuracy) and deployed as a web service.

## Project Set Up and Installation

- Created new workspace called udacity-capstone
- Created new compute instance (DS-3) to be used by workspace/notebooks
- Forked nd00333-capstone project to my github from Udacity's instance
- Uploaded code to my workspace
- Imported dataset in the workspace (penguins.csv)
- Train model using AutoML
- Train model using HyperDrive
- Compare model performance - AutoML vs HyperDrive
- Select the best performing model via the comparison
- Deploy the best performing model as a web service
- Test the model
- Added screenshots throughout the project
- Editing/updated readme.md

![overview](./capstone-diagram.png)

## Dataset - penguins.csv

This project uses the Palmer Archipelago penguin data from Kaggle -> https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data

![dataset](./screenshots/penguin.png)

### Overview

Palmer Archipelago (Antarctica) penguin data

Data collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
