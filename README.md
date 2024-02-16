# Deep Classifier Mimicry without Data Access

Code for our paper **Deep Classifier Mimicry without Data Access**; Steven Braun, Martin Mundt, and Kristian Kersting; _International Conference on Artificial Intelligence and Statistics (AISTATS), 2024_.

**Abstract**:
Access to pre-trained models has recently emerged as a standard across numerous machine learning domains. Unfortunately, access to the original data the models were trained on may not equally be granted. This makes it tremendously challenging to fine-tune, compress models, adapt continually, or to do any other type of data-driven update. We posit that original data access may however not be required. Specifically, we propose Contrastive Abductive Knowledge Extraction (CAKE), a model-agnostic knowledge distillation procedure that mimics deep classifiers without access to the original data. To this end, CAKE generates pairs of noisy synthetic samples and diffuses them contrastively toward a model's decision boundary. We empirically corroborate CAKE's effectiveness using several benchmark datasets and various architectural choices, paving the way for broad application. 

## Examples
Run CAKE on MNIST:
```shell
python src/main.py experiment=mnist-cnn
```

We use hydra's multirun feature enabled with the `-m/--multirun` flag and can specify multiple values for specific
configurations (e.g. `sampling.noise` as below). 

```shell
python src/main.py -m sampling.noise=1e-3,1e-2,1e-1
```

Configurations are found as YAML in `conf/config.yaml` and can be replaced by commandline specifications
```shell
python src/main.py sampling.num_steps=1000 student.epochs=10
```

To print the current configuration, run
```shell
python src/main.py --cfg job
```

Enable WandB logs:
```shell
python src/main.py env.wandb=true ...
```


## Structure
- `conf`: Configuration files
- `conf/experiment`: Specific experiment configuration overrides
- `src`: Python code

## Installation

We made sure to capture all version specific dependencies in `requirements.txt`:

``` sh
pip install -r requirements.txt
```

Tested with Python 3.10.13.

## Major Libraries
- PyTorch: Autograd and Networks
- Lightning: ML Pipeline
- timm: Vision models
- Hydra: Configuration
- WandB: Logging


# Cite
``` bibtex
@misc{braun2024deep,
      title={Deep Classifier Mimicry without Data Access}, 
      author={Steven Braun and Martin Mundt and Kristian Kersting},
      year={2024},
      journal={International Conference on Artificial Intelligence and Statistics (AISTATS)}
}
```
