# [Official Release] MUSE: Illustrating Textual Attributes by Portrait Generation

Resources are included in two folders: code and additional_resources, respectively. The code folder contains the codes for training and testing as well as the MUSE dataset. The additional resources contains the portrait data from wikiart and wikidata. See the Readme files in each folder for details.

## Codes and MUSE Dataset
Codes for training and testing and the MUSE dataset including the photo-portrait pairs for training and testing, and their attribute annotations.

## Additional Resources
Portraits from wikiart and wikidata with image download links and additional information

```bash
├── additional_resources
│   ├── README.md
│   ├── wikiart.json
│   ├── wikidata.json
├── code
│   └── dataset
│   │   └── photo2portrait
│   │       ├── annotation
│   │       ├── attribute_embeddings.pkl
│   │       ├── glove_embeddings.pkl
│   │       ├── test
│   │       │   ├── A
│   │       │   └── B
│   │       └── train
│   │           ├── A
│   │           └── B
│   ├── demo.py
│   ├── discriminator.py
│   ├── generator.py
│   └── output
│       └── models
│   └── README.md
│   └── test_options.py
│   └── train_options.py
│   └── train.py
│   └── util
│       └── __init__.py
│       └── get_loss.py
│       └── load_test_data.py
│       └── load_train_data.py
│       └── logger.py
│       └── model_func.py
└── README.md
```
