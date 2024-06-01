# mamba-music-generation

## Tools used in this project
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://mathdatasimplified.com/stop-hard-coding-in-a-data-science-project-use-configuration-files-instead/)
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting

* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management - [article](https://mathdatasimplified.com/poetry-a-better-way-to-manage-python-dependencies/)


## Project Structure

```bash
.
├── config                      
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model1.yaml             # First variation of parameters to train model
│   │   └── model2.yaml             # Second variation of parameters to train model
│   └── process                     # Configurations for processing data
│       ├── process1.yaml           # First variation of parameters to process data
│       └── process2.yaml           # Second variation of parameters to process data
├── data            
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   └── raw                         # raw data
├── docs                            # documentation for your project
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
│   ├── process.py                  # process data before training model
│   └── train_model.py              # train model
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    ├── test_process.py             # test functions for process.py
    └── test_train_model.py         # test functions for train_model.py
```

## Set up the environemnt with TLDR bash script
1. Install dependencies and download data:
```bash
bash vast_ai_setup.sh
```

2. Login to wandb:
```bash
wandb login
```

3. Test the environment:
```bash
python src/test_env.py
```

## Set up the environment - detailed instructions


1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Activate the virtual environment:
```bash
poetry shell
```
3. Install dependencies:
- To install all dependencies from pyproject.toml, run:
```bash
pip install --upgrade pip setuptools wheel
poetry install
```
- To install only production dependencies, run:
```bash
poetry install --only main
```
- To install a new package, run:
```bash
poetry add <package-name>
```

## Set up wandb
After creating an account on [wandb](https://wandb.ai/site), creating a project, and setting up the variables in the *config/main.yaml*, then run in the terminal:
```bash
wandb login
```

# Installation verification!

After these commands paste:
```bash
python src/test_env.py
```

If it won't work try with installing following dependencies through pip:

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install causal_conv1d==1.2.*
pip install mamba-ssm==1.2.0.post1
```

In case the above does not work, please make sure you have cuda devel toolkit installed.
## Before all commit

To make shure that your code fulfill all necessary from yapf and pylint paste whi before all commits and correct them

```bash
./yapf-fix.sh
```
```bash
./pylint.sh
```
## Train model
1. Download data:
```bash
python src/download_data.py
```

2. Train model:
```bash
python src/train_model.py > "logs/output_$(date +'%Y-%m-%d_%H-%M-%S').log" 2>&1
```

## View and alter configurations
To view the configurations associated with a Pythons script, run the following command:
```bash
python src/process.py --help
```
Output:
```yaml
process is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

model: model1, model2
process: process1, process2


== Config ==
Override anything in the config (foo.bar=value)

process:
  use_columns:
  - col1
  - col2
model:
  name: model1
data:
  raw: data/raw/sample.csv
  processed: data/processed/processed.csv
  final: data/final/final.csv
```

To alter the configurations associated with a Python script from the command line, run the following:
```bash
python src/process.py data.raw=sample2.csv
```

## Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```

# Inference
After training, you can fined models in `models/` directory.

1. Choose a model from `models/` directory.
2. Make sure the config in `config/mamba_model.yaml` is the same as the model you want to use.
3. Run the inference:
```bash
python src/run_inference.py
```


# WIMU projekt
## Design proposal
### Temat projektu
***Symbolic music generation using Mamba architecture***

Przez ostatnie lata wiele wysiłku zostało włożone w to, aby uczynić Transformery coraz bardziej wydajnymi. Jednak od stosunkowo niedawna odczuwają one coraz większą konkurencję ze strony modeli oparytch o architekturę SSM (State Space Models), które dorównją, a nawet pokonują je w wielu zadaniach. Najnowszym i jak dotąd najlepszym modelem SSM jest Mamba. Nie znaleźliśmy żadnych badań dotyczących wykorzystania tej architektury do generowania muzyki w formacie symbolicznym. W związku z tym chcielibyśmy sami sprawdzić jak Mamba sprawdza się w tym zadaniu.
Perspektywy wydają się bardzo obiecujące ze względu na specyfikację tej architektury, której przewaga nad klasycznymi transformerami uwypukla sie w przypadku coraz to dłuższych sekwencji. Dlatego muzyka wydaje się idealnym polem do zbadania działania Mamby.

Projekt ma na celu zbadanie możliwości generowania muzyki symbolicznej przy użyciu architektury Mamba. Docelowo planujemy generować muzykę w formacie MIDI, lecz jest możliwe że ostatecznie skupimy się na formacie ABC lub MusicXML.

### Harmonogram
**UWAGA:** Harmonogram będzie zmieniany dynamicznie wraz z postępem prac.

| Tydzień         | TODO |
|-----------------|------|
| 19 Feb - 25 Feb |==początek semestru==|
| 26 Feb - 03 Mar |wybór tematu|
| 04 Mar - 10 Mar |konsultacja tematu|
| 11 Mar - 17 Mar |design proposal|
| 18 Mar - 24 Mar |przygotowanie repo, MLOps|
| 25 Mar - 31 Mar |==święta==|
| 01 Apr - 07 Apr |zaznajomienie się z architekturą i rozplanowanie implementacji|
| 08 Apr - 14 Apr |rozpoczęcie implementacji|
| 15 Apr - 21 Apr |wybranie formatu danych, datasetu i sposobu tokenizacji|
| 22 Apr - 28 Apr |trenowanie, ewaluacja i modyfikacja 1|
| 29 Apr - 05 May |==majówka==|
| 06 May - 12 May |trenowanie, ewaluacja i modyfikacja 2|
| 13 May - 19 May |trenowanie, ewaluacja i modyfikacja 3|
| 20 May - 26 May |analiza wyników i przygotowanie artykułu|
| 27 May - 02 Jun |**zgłosznie artykułu (deadline)**|
| 03 Jun - 09 Jun |tydzień na ewentualne dokończenie prac|
| 10 Jun - 16 Jun |==koniec semestru==|


### Planowany zakres eksperymentów
1. Trenowanie architektury Mamba do generowania muzyki w formacie symbolicznym (MIDI, ABC albo MusicXML).
2. Ewaluacja wyników (głównie poprzez ankietyzacje z testem statystycznym).
    - Porównanie muzyki wygenerowanej do prawdziwej oraz wygenerowanej przez inne modele.
    - Przeprowadzenie analizy Mean Opinion Score (MOS).
    - Metryki.
3. (Opcjonalne) Porównanie z transformerem.
4. (Opcjonalnie) Dodanie warunkowania tekstem.
5. Opisanie następnych kroków i możliwych opcji rozwoju.

### Planowany stack technologiczny
1. Python 3
    - autoformatter: black
    - linter: ruff
    - Środowisko wirtualne: Pipenv lub Pip + venv
    - struktura projektu: cookiecutter
2. PyTorch
    - Tensorboard / MLFlow / wandb
    - PyTorch Lightning
3. MidiTok
4. MusPy
5. pretty_midi
6. Git
7. GitHub 
8. Huggingface


## Bibliografia
- GU, Albert; DAO, Tri. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
- GU, Albert, et al. Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems, 2021, 34: 572-585.
- GU, Albert; GOEL, Karan; RÉ, Christopher. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.
- SMITH, Jimmy TH; WARRINGTON, Andrew; LINDERMAN, Scott W. Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933, 2022.
- VINAY, Ashvala; LERCH, Alexander. Evaluating generative audio systems and their metrics. arXiv preprint arXiv:2209.00130, 2022.
- YANG, Li-Chia; LERCH, Alexander. On the evaluation of generative models in music. Neural Computing and Applications, 2020, 32.9: 4773-4784.
- FRADET, Nathan, et al. MidiTok: A python package for MIDI file tokenization. arXiv preprint arXiv:2310.17202, 2023.
- Peiling Lu, Xin Xu, Chenfei Kang, Botao Yu, Chengyi Xing, Xu Tan, Jiang Bian, MuseCoco: Generating Symbolic Music from Text, arXiv preprint arXiv:2306.00110.pdf, 2023


### Tabela z analizą źródeł
[Link](docs/sources_analysis_table.md)

### Omówienie architektury Mamba
[Link](docs/mamba.md)