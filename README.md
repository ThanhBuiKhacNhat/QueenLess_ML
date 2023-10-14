# Queenless_Monitoring
> Queenless Monitoring: new MFCCs approach for feature extraction and hyperparameters tuning models in detecting the absence of queens bee by sound

## Introduction
This repository supports cropping and extracting audio features from the dataset and then tuning hyperparameters for several machine learning models to evaluate the performance of each model. The only thing you need to do is install the required libraries and click 'run'.

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

After downloading, extract the zip file and put it in the project directory. The directory structure must look like this:

``` bash
Queenless_Monitoring
├── dataset
│   ├── NoQueen
│   ├── NoQueen_Kit
│   ├── NoQueen_test
│   ├── Queen_Kit
│   ├── Queen_Record
│   └── Queen_test
├── RUNME.py
├── README.md
└── requirements.txt
```

Four folders: `NoQueen`, `NoQueen_Kit`, `Queen_Kit`, `Queen_Record` are in the zip file in the above link, but you have to create two folders `NoQueen_test` and `Queen_test` by yourself. Note that the name must be correct (case-sensitive). The content of each folder is as follows:

- Select one file in `NoQueen` or `NoQueen_Kit` and move it to folder `NoQueen_test` (meaning that file cannot exist in folders `NoQueen` and `NoQueen_Kit` anymore).
- Select one file in `Queen_Kit` or `Queen_Record` and move it to folder `Queen_test` (meaning that file cannot exist in folders `Queen_Kit` and `NoQueen_Record` anymore).

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

Note: If you want to deactivate the virtual environment, just type `deactivate` in your terminal.

### Install required libraries

```bash
pip install -r requirements.txt
```

### Run the project

Note: You should change the **SIGNATURE** variable in the file `RUNME.py` before running it. The **SIGNATURE** is the name of the folder containing csv files that are extracted from audios and the results of your experiment.

```bash
python3 RUNME.py
```

Or you can also click the 'run' button in your IDE.

Note: If you want to run the project with the 'run' button, you should specify the environment in your IDE to the virtual environment you have just created.

### Update the new version

To update to a new version of the project, you run the script:

```bash
git pull
```
