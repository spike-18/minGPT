# Palindromes

Test GPT on task of classification whether input digit sequence is a palindrome or not.

## Project structure

```bash

palindromes/
├── checkPal.py           # Implementation of dataset, training routine
├── testModel.py          # Module to test model on user input
├── utils.py              # Utils
│
└── README.md             # This file
```

(If some files/directories are missing in your checkout, use the above structure as intended for organizing code.)

## Usage

Specify configuration in checkPal.py module

```python

    ...

    C.model.model_type = 'gpt-micro'
    C.model.num_classes = 2     # Since we predict class palindrome / not palindrome
    C.model.block_size = 50     # 50-digit sequences max input

```

(Optional) Train the model:

``` bash

cd palindromes
python checkPal.py
```

Run test of the model:

``` bash

python testModel.py out/palindromes/model.pt
```

## Results

As a result of my experiments a tiny model 'GPT-micro' of 0.8M params can successefully classify palindromes of length 50 digits.
