Implement a Complete MLOps Pipeline
Project Overview
In this project, you will take a machine learning model from a raw dataset all the way through a production-ready MLOps pipeline. You will apply everything you learned in this sprint: version control with Git and DVC, experiment tracking with MLflow, automated testing with pytest, CI/CD with GitHub Actions, and drift monitoring with Evidently.

You will work with a dataset of your choosing (guidelines below) and build a complete, end-to-end system that a real ML team could use as the foundation for production operations. This is not a modeling project. The model itself can be simple. What matters is the infrastructure, the practices, and the automation surrounding the model.

Dataset Requirements
You may choose any tabular classification or regression dataset that meets the following criteria:

The dataset must have at least 1,000 rows and at least 8 features. It must include a mix of numeric and categorical columns. It must include some missing values (or you must simulate missing values if the original dataset is fully clean). 

Here are some suggested datasets if you do not have one in mind:

Heart Disease Prediction from the UCI Machine Learning Repository. Classification task with numeric and categorical features, missing values, and a clear target variable.

Employee Attrition from Kaggle (IBM HR Analytics). Classification task predicting whether employees leave, with a good mix of feature types.

California Housing from scikit-learn. Regression task predicting house prices, widely available, easy to work with.

Student Dropout dataset used throughout this sprint. You are welcome to reuse it, but your project code must be written from scratch, not copied from the lesson exercises.

If you choose a different dataset, include a brief note in your README explaining where it comes from and what the prediction task is.

Project Requirements
Your project must include all of the components described below. Each component maps to a chapter of the sprint and demonstrates that you can apply the concepts in an integrated workflow.

1. Repository Structure and Version Control (Git + DVC)
Your project must be a public GitHub repository with a clean, organized structure. At minimum, your repository should contain the following:

A src/ directory with your Python source code, separated into logical modules (not one giant script). At minimum, you should have separate files for data preprocessing, model training, and evaluation.

A configs/ directory with at least one YAML configuration file that defines your training hyperparameters, file paths, and any other settings. Your training script must read from this config file rather than using hardcoded values.

A tests/ directory with your pytest test suite.

A .github/workflows/ directory with your GitHub Actions workflow file.

A requirements.txt file with pinned dependency versions.

A .gitignore file that properly excludes data files, model artifacts, Python caches, and other files that do not belong in Git.

A README.md that explains what the project is, how to set it up, how to run training, and how to run tests.

DVC must be initialized and configured with a remote (a local remote is acceptable). Your training dataset must be tracked with DVC, not committed to Git. The .dvc pointer files must be committed to Git. A grader should be able to clone your repository, run dvc pull, and have the dataset available.

2. Experiment Tracking (MLflow)
Your training script must integrate MLflow for experiment tracking. Every training run must log the following:

All hyperparameters from your config file (model type, learning rate, number of estimators, or whatever is relevant to your model).

The data version being used (the DVC file hash or a descriptive label).

All evaluation metrics (at minimum: the primary metric for your task plus two additional metrics).

The trained model as an MLflow artifact using the appropriate mlflow.<framework>.log_model() call.

You must run at least five experiments with different configurations (different hyperparameters, different feature sets, or different model types) and all five must be visible in your MLflow tracking.

You must include a script or notebook called compare_experiments.py (or similar) that uses mlflow.search_runs() to programmatically query your experiments and identify the best run based on your primary metric.

3. Testing (pytest)
Your test suite must include tests at three levels:

Unit tests for preprocessing functions. At minimum, write tests that verify: a function correctly handles missing values, a function correctly encodes categorical variables, a function does not modify the original dataframe, and a function raises appropriate errors for invalid input. You need at least six unit tests total.

Data validation tests. Write at least three tests that verify properties of your dataset: expected columns are present, the target variable contains only expected values (for classification) or is within a reasonable range (for regression), and numeric features are within expected ranges. These tests should load the actual dataset (or a sample of it) and verify its properties.

Model validation tests. Write at least two tests that train a model on a small sample and verify: the model produces predictions of the correct type and shape, and the model achieves a minimum performance threshold on a known test set.

All tests must pass when a grader runs pytest tests/ -v from the project root.

4. CI/CD Pipeline (GitHub Actions)
Your repository must include a GitHub Actions workflow file that defines an automated pipeline. The pipeline must:

Trigger on pushes to the main branch and on pull requests targeting main.

Have at least two jobs: one for testing and one for training. The training job must depend on the test job passing.

The test job must install dependencies and run your full pytest suite.

The training job must install dependencies, run your training script, and verify that the model meets minimum performance thresholds (the script should exit with a non-zero code if thresholds are not met).

The pipeline must have run at least once successfully. Graders will check the Actions tab of your repository for a green pipeline run.

5. Drift Monitoring (Evidently)
Your project must include a drift monitoring script that compares a reference dataset against simulated production data.

Create a script called src/monitor_drift.py (or similar) that:

Loads your training data as the reference distribution.

Loads or generates a "production" dataset that simulates data arriving after deployment. You may either split your original dataset to simulate production data, or generate synthetic drifted data similar to what we did in the hands-on lesson.

Runs Evidently drift detection on all features.

Prints a summary showing which features drifted and the overall drift share.

Saves an HTML report to a reports/ directory.

Exits with code 1 if drift exceeds a configurable threshold.

You must also include a brief written analysis (in your README or in a separate MONITORING.md file) that answers these three questions: Which features showed drift and why? Would this drift likely affect model performance? What action would you recommend (retrain, investigate, or continue monitoring)?

Deliverables
Your final submission is a link to your public GitHub repository. The repository must contain everything described above. Graders will evaluate your project by cloning the repository, reading the README, inspecting the code, running the tests, reviewing the GitHub Actions history, and reading your drift analysis.

Specifically, graders will receive your repository URL and follow these steps:

Step 1: Clone the repository and read the README. Is the project clearly explained? Are setup instructions provided?

Step 2: Inspect the folder structure. Is it organized according to best practices?

Step 3: Check .gitignore and verify that data and model files are not committed to Git.

Step 4: Check for DVC initialization and pointer files.

Step 5: Read the config YAML file and verify the training script reads from it.

Step 6: Run pytest tests/ -v and verify all tests pass.

Step 7: Read through the test code and evaluate test quality and coverage.

Step 8: Check the GitHub Actions tab for at least one successful pipeline run.

Step 9: Read the workflow YAML file and verify it meets the requirements.

Step 10: Check for MLflow experiment runs (at minimum by reading the training script and experiment comparison script to verify MLflow integration is correct).

Step 11: Run the drift monitoring script and review the output and HTML report.

Step 12: Read the drift analysis writeup.