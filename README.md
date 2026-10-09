# MLOps on AWS

Coursework repo for Udacity's Machine Learning Engineer Nanodegree, Course 2: **Developing Your First ML Workflow**. The notebooks walk through building, deploying, and monitoring machine learning workflows on AWS, mostly with Amazon SageMaker.

The exercises use a running example called **HelloBlaze**: a BlazingText sentiment model trained on Amazon product reviews to predict whether a review is helpful. Each exercise and its worked solution are separate notebooks.

## Topics covered

- **lesson2: SageMaker essentials.** Training jobs, real-time endpoints, batch transform, and processing jobs. Exercises 1 to 4 build the HelloBlaze pipeline piece by piece; Exercise 5 ties it together with an end-to-end XGBoost sentiment classifier on the IMDb movie review dataset. The `demo` folder has the short lecture demos (XGBoost training, endpoints, batch transform, processing jobs, plus a CLI training-job script).
- **lesson3: Workflows.** AWS Lambda (writing and invoking functions, S3 triggers), Step Functions (chaining a SageMaker processing step and training step), and launching a Step Function from a Lambda function. `HelloBlazePreprocess.py` and `HelloBlazePreprocessLambda.py` are the preprocessing scripts used by the processing jobs and Lambda functions.
- **lesson4: Feature Store, Model Monitor, and Clarify.** Creating feature groups and ingesting data with SageMaker Feature Store, configuring data capture and monitoring schedules with Model Monitor, and explainability monitoring with SageMaker Clarify (SHAP). Includes `demos.ipynb`, plus exercise starters and worked solutions.
- **project: Deploy and monitor an image classification workflow.** Downloads CIFAR-100, filters it to bicycles vs motorcycles, trains SageMaker's built-in image classification algorithm, deploys the model with Model Monitor data capture, and drafts the Lambda plus Step Function workflow for low-confidence filtering. `test.lst` and `train.lst` are the image manifest files the training job consumes.

## AWS prerequisites

These notebooks call live AWS services (SageMaker, S3, Lambda, Step Functions, SageMaker Feature Store) and **require AWS credentials** with the appropriate IAM permissions. Most notebooks were written for a SageMaker notebook instance, where `sagemaker.get_execution_role()` resolves the execution role automatically.

Things to know before running:

- Running the notebooks provisions billable AWS resources (training instances, endpoints, processing jobs). Delete endpoints, monitoring schedules, and workflows when you are done. Several notebooks end with cleanup cells; run them.
- Notebooks that upload data expect you to replace placeholder bucket names (marked `CHANGE THIS` or `FILL_THIS_IN`) with a bucket you own. For the Lambda preprocessing script, set the `PREPROCESS_BUCKET` and `PREPROCESS_PREFIX` environment variables.
- Some notebooks pin a region (for example `us-west-2`) or reference specific S3 paths from the original course; update those to your own account and region.

## How to run

1. Create a SageMaker notebook instance (the `ml.t3.medium` size is enough for everything except training, where the notebooks pick their own training instances), or run locally with AWS credentials configured.
2. Install dependencies: `pip install -r requirements.txt`. A SageMaker notebook instance already ships most of these.
3. Open a lesson folder and run the notebooks in exercise order. Each exercise notebook has a matching `Solution` notebook with the completed code.
4. For Lesson 2, Exercise 5 and the project, the notebooks download their datasets automatically (IMDb reviews and CIFAR-100).

## What you learn

- How to launch and configure SageMaker training jobs, endpoints, batch transform jobs, and processing jobs, from both the console and the Python SDK.
- How to preprocess data for SageMaker's built-in algorithms (BlazingText, XGBoost, image classification), including the label-first CSV and JSON-lines formats they expect.
- How to build serverless ML workflows: Lambda functions triggered by S3 uploads, Step Functions chaining SageMaker jobs, and Lambda functions that launch state machines.
- How to monitor deployed models in production: SageMaker Feature Store for consistent features, Model Monitor for data drift with scheduled baselines, and Clarify for explainability.
- How to put it together: an image classification service with data capture, low-confidence filtering via Lambda, and orchestration via Step Functions.

## Repo layout

```
lesson2/   SageMaker essentials: exercises, solutions, lecture demos
lesson3/   Lambda and Step Functions: exercises, solutions, lecture demos, preprocessing scripts
lesson4/   Feature Store, Model Monitor, Clarify: demos, exercise starters, solutions
project/   Final project: image classification workflow notebook and manifests
```
