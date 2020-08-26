''' Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0 '''

# Seed random number generators to ensure identical results for everyone
from numpy.random import seed
seed(42)
from tensorflow import random
random.set_seed(42)
########################################################################

import argparse
import os
import numpy as np
import json
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def model(x_train, y_train, x_test, y_test, args):
    """Generate a simple model"""
    model = Sequential([
                Dense(args.l1_size, activation=args.l1_activation, kernel_initializer='normal'),
                Dense(args.l2_size, activation=args.l2_activation, kernel_initializer='normal'),
                Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss='mean_squared_logarithmic_error',
                  metrics=['mean_squared_logarithmic_error'])
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1)
    model.evaluate(x_test,y_test,verbose=1)

    return model


def _load_training_data(base_dir):
    """Load the abalone training data"""
    x_train = (pd.read_csv(os.path.join(base_dir, 'abalone_train_features.csv'))).to_numpy()
    y_train = (pd.read_csv(os.path.join(base_dir, 'abalone_train_labels.csv'))).to_numpy()
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load the abalone testing data"""
    x_test = (pd.read_csv(os.path.join(base_dir, 'abalone_test_features.csv'))).to_numpy()
    y_test = (pd.read_csv(os.path.join(base_dir, 'abalone_test_labels.csv'))).to_numpy()
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--l1_size', type=int, default=10)
    parser.add_argument('--l1_activation', type=str, default='relu')
    parser.add_argument('--l2_size', type=int, default=10)
    parser.add_argument('--l2_activation', type=str, default='relu')

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    abalone_regressor = model(train_data, train_labels, eval_data, eval_labels, args)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory
        abalone_regressor.save(args.sm_model_dir, 'abalone_model.h5')