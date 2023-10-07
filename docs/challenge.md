Software Engineer (ML & LLMs) Challenge Solution

## Overview
The challenge solution is hosted on a github repo. The repo is public and can be found [here](https://github.com/giovannicarlos77/latam_task).

### Solution explanations
- The model selected was Logistic Regression due to:
- Better metrics than XGBoost in classification_report
- Normally demands less computational resources
- Easier to understand and explain if needed

## Usage
To use the solution, you can either:
- Clone the repo and run it locally
  - use git clone on the repo main branch
  - run command: `pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt`
  - run `run_challenge.py` in root directory

## API Documentation
To see the API documentation, you can either:
- locally: run the application as described above and go to the http://localhost:8000/docs

## Notes
- I not deployed in the GCP because its having some issues but in localhost works perfectly.