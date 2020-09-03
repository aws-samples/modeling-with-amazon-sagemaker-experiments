# Streamline Modeling With Amazon SageMaker Studio and Amazon Experiments SDK

Modeling phase is a highly iterative process in Machine Learning projects, where Data Scientists experiment with various data pre-processing and feature engineering strategies, intertwined with different model architectures, which are then trained with disparate sets of hyperparameter values. This highly iterative process, with many moving parts, can, over time, manifest into a tremendous headache in terms of keeping track of all design decisions applied in each iteration and how the training and evaluation metrics of each iteration compare to the previous versions of the model.

While your head may be spinning by now, fear not! Amazon SageMaker has a solution!

The provided `modeling-with-amazon-sagemaker-experiments.ipynb` Jupyter notebook walks you through an end-to-end example of utilizing [Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio.html) and [Amazon SageMaker Experiments SDK](https://sagemaker-experiments.readthedocs.io/en/latest/), to organize, track, visualize and compare our iterative experimentation with a Keras model. While this example is specific to Keras framework, the same approach can be extended to other Deep Learning frameworks and Machine Learning algorithms.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

