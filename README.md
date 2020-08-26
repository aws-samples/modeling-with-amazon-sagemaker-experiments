# Streamline Modeling With SageMaker Studio and Experiments SDK

In most Machine Learning projects, modeling phase tends to be a highly iterative process, where Data Scientists explore various data preprocessing and feature engineering approaches, as well as, a variety of model architectures with varying set of hyperparameters. This highly iterative process, with many moving parts, can, over time, manifest into a tremendous challenge of keeping track of all design decisions applied in each iteration, and how the performance metrics of each iteration compare to the other versions of the solution.

The provided `modeling-with-amazon-sagemaker-experiments.ipynb` Jupyter notebook walks through an end-to-end example of how SageMaker Studio can effectively leverage SageMaker Experiments SDK to organize, track, visualize and compare our iterative work during development of a Keras model, trained to predict the age of an abalone, a sea snail, based on a set of features that describe it. While this example is specific to Keras, the same approach can be extended to other Machine Learning frameworks and algorithms.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

