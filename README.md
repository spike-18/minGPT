# minGPT

A minimal, educational GPT-style transformer implementation for experimenting with small-scale language modeling experiments.

## Project structure

```bash

minGPT/
├── projects/                 # Projects
│   └── palindromes/
├── src/mingpt/               # Transformer package
│   ├── bpe.py                # Basic tokenizer
│   ├── model.py              # GPT model implementation
│   ├── trainer.py            # Utilitary module for training routine
│   └── utils.py              # Utils
├── pyproject.toml            # Package configuration file
└── README.md                 # This file
```

(If some files/directories are missing in your checkout, use the above structure as intended for organizing code.)

## Installation

```bash

git clone https://github.com/spike-18/minGPT.git
cd minGPT
pip install -e .
```

## Usage

Here's how you'd instantiate a GPT-2 (124M param version):

```python

from mingpt.model import GPT
model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
model = GPT(model_config)
```

And here's how you'd train it:

``` python

# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
train_dataset = YourDataset()

from mingpt.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # many possible options, see the file
train_config.max_iters = 1000
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.run()
```

## License

- MIT
