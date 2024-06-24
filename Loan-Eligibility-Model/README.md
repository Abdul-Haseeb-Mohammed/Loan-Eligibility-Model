Loan Eligibility Model
==============================

Classification model to predict eligibility of a loan applicant

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
<h1>Loan Eligibility Prediction</h1>
    <p>
        Loans form an integral part of banking operations. However, not all loans are returned and hence it is important for a bank to closely monitor its loan applications. This case study is an analysis of the German Credit data. It contains details of 614 loan applicants with 13 attributes and the classification whether an applicant was granted loan or denied loan.
    </p>

<h2>Your Role</h2>
    <p>
        Using the available dataset, train a classification model to predict whether an applicant should be given a loan.
    </p>

<h2>Goal</h2>
    <p>
        Build a model to predict loan eligibility with an average accuracy of more than 76%.
    </p>

<h2>Specifics</h2>
    <ul>
        <li><strong>Machine Learning task:</strong> Classification model</li>
        <li><strong>Target variable:</strong> Loan_Status</li>
        <li><strong>Input variables:</strong> Refer to data dictionary below</li>
        <li><strong>Success Criteria:</strong> Accuracy of 76% and above</li>
    </ul>

<h2>Data Dictionary</h2>
    <ul>
        <li><strong>Loan_ID:</strong> Applicant ID</li>
        <li><strong>Gender:</strong> Gender of the applicant Male/Female</li>
        <li><strong>Married:</strong> Marital status of the applicant</li>
        <li><strong>Dependents:</strong> Number of dependents the applicant has</li>
        <li><strong>Education:</strong> Highest level of education</li>
        <li><strong>Self_Employed:</strong> Whether self-employed Yes/No</li>
        <li><strong>ApplicantIncome:</strong> Income of the applicant</li>
        <li><strong>CoapplicantIncome:</strong> Income of the co-applicant</li>
        <li><strong>LoanAmount:</strong> Loan amount requested</li>
        <li><strong>Loan_Amount_Term:</strong> Term of the loan</li>
        <li><strong>Credit_History:</strong> Whether applicant has a credit history</li>
        <li><strong>Property_Area:</strong> Current property location</li>
        <li><strong>Loan_Approved:</strong> Loan approved yes/no</li>
    </ul>

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
