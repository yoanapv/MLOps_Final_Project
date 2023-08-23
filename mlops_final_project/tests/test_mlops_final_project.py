#!/usr/bin/env python

"""Tests for `mlops_finalproject` package."""

import os
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from mlops_finalproject.load.load_data import DataRetriever
from mlops_finalproject.preprocess.preprocess_data import CategoricalImputer
from mlops_finalproject.preprocess.preprocess_data import OrderingFeatures

def does_csv_file_exist(file_path):
    """
    Check if a CSV file exists at the specified path.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)

def test_csv_file_existence():
    """
    Test case to check if the CSV file exists.
    """
    # Provide the path to your CSV file that needs to be tested
    os.chdir('/Users/norma.perez/Documents/GitHub/MLOps_FinalProject/mlops_finalproject/mlops_finalproject')
    csv_file_path = "./data/retrieved_data.csv"
    
    DATASETS_DIR = './data/'
    
    URL = '/Users/norma.perez/Documents/GitHub/MLOps_FinalProject/mlops_finalproject/mlops_finalproject/data/Hotel_Reservations.csv'
    data_retriever = DataRetriever(URL, DATASETS_DIR)
    data_retriever.retrieve_data()

    # Call the function to check if the CSV file exists
    file_exists = does_csv_file_exist(csv_file_path)

    # Use Pytest's assert statement to check if the file exists
    assert file_exists == True, f"The CSV file at '{csv_file_path}' does not exist."

def test_categorical_imputer():
    # Create a sample DataFrame
    data = pd.DataFrame({
        'type_of_meal_plan': ['Meal Plan 1', None, 'Meal Plan 2', 'Meal Plan 3'],
        'room_type_reserved': [None, 'Room_Type 3', 'Room_Type 1', 'Room_Type 2'],
        'market_segment_type': ['Offline', 'Offline', None, 'Online']
    })

    # Instantiate the custom transformer
    imputer = CategoricalImputer(variables=['type_of_meal_plan', 'room_type_reserved','market_segment_type'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('imputer', imputer),
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(data)

    # Check if the transformation was done correctly
    assert X_transformed['type_of_meal_plan'].isnull().sum() == 0
    assert X_transformed['room_type_reserved'].isnull().sum() == 0
    assert X_transformed['market_segment_type'].isnull().sum() == 0
    assert 'Missing' in X_transformed['type_of_meal_plan'].values
    assert 'Missing' in X_transformed['room_type_reserved'].values
    assert 'Missing' in X_transformed['market_segment_type'].values

def test_ordering_features():
    # Create a sample DataFrame
    data = pd.DataFrame({
        'Booking_ID': ['INN00001', 'INN00002', 'INN00003', 'INN00004'],
        'type_of_meal_plan': ['Meal Plan 1', None, 'Meal Plan 2', 'Meal Plan 3'],
        'room_type_reserved': [None, 'Room_Type 3', 'Room_Type 1', 'Room_Type 2'],
        'market_segment_type': ['Offline', 'Offline', None, 'Online']
    })

    # Instantiate the custom transformer
    orderer = OrderingFeatures()

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('orderer', orderer),
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(data)

    # Check if the transformation was done correctly
    assert list(X_transformed.columns) == ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

def test_model_and_pipeline_saved():

    TRAINED_MODEL_DIR = '/Users/norma.perez/Documents/GitHub/MLOps_FinalProject/mlops_finalproject/mlops_finalproject/models/'
    MODEL_SAVE_FILE = 'extra_trees_classifier_model_output.pkl'
    PIPELINE_SAVE_FILE = 'extra_trees_classifier_pipeline_output.pkl'
    # Define the paths to the saved model and pipeline
    model_save_path = TRAINED_MODEL_DIR + MODEL_SAVE_FILE
    pipeline_save_path = TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE

    # Check if the model and pipeline files exist
    assert os.path.isfile(model_save_path), f"Model not saved in {model_save_path}"
    assert os.path.isfile(pipeline_save_path), f"Pipeline not saved in {pipeline_save_path}"
