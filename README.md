# Queenless_Monitoring
> Queenless Monitoring: new MFCCs approach for feature extraction and hyperparameters tuning models in detecting the absence of queens bee by sound

## Introduction
This repository supports splitting, extracting audio features, feature selection from the dataset and tuning hyperparameters for several machine learning models to evaluate the performance of each model. The only thing you need to do is install the required libraries and click 'run'.

## Guide

Here is a detailed step-by-step instruction to run this project on your local machine.

### Clone the project

Through HTTPS:

```bash
git clone https://github.com/hoanghai1803/Queenless_Monitoring.git
```

Through SSH:

```bash
git clone git@github.com:hoanghai1803/Queenless_Monitoring.git
```

Or you can also download the zip file.

### Download the dataset

You can download the dataset and paper here: [Queenless Monitoring](https://fptuniversity-my.sharepoint.com/:f:/g/personal/hainhde170683_fpt_edu_vn/EqsOqSf3G0NCmTJzthxiv2YBcqnE6xqg4y0mLNDuOgzopw?e=fGTYwm). 

In addition, you can get the training statistics and the results of the experiments [here](https://docs.google.com/spreadsheets/d/1rsPMrFoV7OyAkBHeVOlhnCXcu6J6t85suTIBVdy06LY/edit#gid=0).

In the above link, you can download the zip file named `Queen_NoQueen.zip` for training and `Queen_NoQueen_Test.zip` for testing.

After downloading, extract the zip file and put it in the project directory. The directory structure must look like this:

``` bash
Queenless_Monitoring
├── dataset
│   ├── NoQueen
│   ├── NoQueen_Kit
│   ├── NoQueen_Test
│   ├── Queen_Kit
│   ├── Queen_Record
│   └── Queen_Test
├── RUNME.py
├── README.md
└── requirements.txt
```

Four folders: `NoQueen`, `NoQueen_Kit`, `Queen_Kit`, `Queen_Record` are in the `Queen_NoQueen.zip` file, but you have to create two folders `NoQueen_test` and `Queen_test` by yourself. Note that the name must be correct (case-sensitive). The content of each folder is as follows:

- Select two files in `Queen_NoQueen_Test.zip` that are labeled have no queen and move it to folder `NoQueen_test`.
- Select four files in `Queen_NoQueen_Test.zip` that are labeled have queen and move it to folder `Queen_test`.

### Create virtual environment

The version of your python should be 3.6 or higher.

```bash
python3 -m venv env
```

### Activate virtual environment

For Linux:

```bash
source env/bin/activate
```

For Windows:

```bash
.\env\Scripts\activate
```

Note:
- In Windows, if you get an error like this: `cannot be loaded because running scripts is disabled on this system`, you can run this command in your terminal: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted` and then type `Y` to accept.
- If you want to deactivate the virtual environment, just type `deactivate` in your terminal.

### Install required libraries

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python3 RUNME.py
```

Or you can also click the 'run' button in your IDE.

Note:
- You should change the **SIGNATURE** variable in the file `RUNME.py` before running it. The **SIGNATURE** is the name of the folder containing csv files that are extracted from audios and the results of your experiment.
- Beside the **SIGNATURE**, there are many environment constants for runtime that you need to specify in the file `RUNME.py`. You can read the comments in the file to understand the meaning of each constant. 
- If you want to run the project with the 'run' button, you should specify the environment in your IDE to the virtual environment you have just created.

### Update the new version

To update to a new version of the project, run the script:

```bash
git pull
```

### Final project structure

The project structure after generating data and training models

``` bash
Queenless_Monitoring
├── dataset
├── env
├── export
│   └── SIGNATURE
│       ├── data
│       │   ├── d_mean.csv
│       │   ├── d_merged.csv
│       │   ├── d_std.csv
│       │   ├── s_mean.csv
│       │   ├── s_merged.csv
│       │   └── s_std.csv
│       ├── models
│       │   ├── decision_tree.pkl
│       │   ├── knn.pkl
│       │   ├── logistic_regression.pkl
│       │   ├── random_forest.pkl
│       │   ├── svm.pkl
│       │   └── xgboost.pkl
│       └── results
│           ├── logs.txt 
│           └── results.csv
├── RUNME.py
├── README.md
└── requirements.txt
```

Have the best experience with our project!
