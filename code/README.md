# Muse: Code and Dataset
The codes for training and testing as well as the MUSE dataset.

## Code
These are the information slots in the code:
- `train.py`: Train the MUSE model using the MUSE train set
- `train_options.py`: Hyperparameters and data paths for training
- `demo.py`: Test the MUSE model using the MUSE test set and save generated portraits at different epochs
- `test_options.py`: Hyperparameters and data paths for testing
- `generator.py`: Attribute-aware UNet
- `discriminator.py`: Discriminator and the attribute classifier
- `util/get_loss.py`: Calculate the attribute classification loss
- `util/load_train_data.py`: Load the photo-portrait pairs and corresponding attributes for training
- `util/load_test_data.py`: Load the photos and corresponding attributes for testing
- `util/model_func.py`: Helper functions
- `util/logger.py`: Display the generated results with Visdom

## MUSE dataset
These are information slots in the MUSE dataset:
|Dataset  | \#portraits|
|:---------:|:---:|
|train  |  3,098|
|test   |     198|

- `train`: A and B folders containing the input photos and golden portraits for training, respectively.
- `test`: A and B folders containing the input photos and golden portraits for testing, respectively.
- `annotation`: Attribute annotations for all the golden portraits
- `attribute_embeddings.pkl`: Attribute embeddings trained on the portraits in additional resources to initialize the embedding layer.
- `glove_embeddings.pkl`: Attribute embeddings from Glove to initialize the embedding layer.
