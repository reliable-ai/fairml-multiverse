# Multiverse Analysis for Algorithmic Fairness

This repository contains the code and data for the paper [One Model Many Scores: Using Multiverse Analysis to Prevent Fairness Hacking and Evaluate the Influence of Model Design Decisions](https://doi.org/10.1145/3630106.3658974) by Jan Simson, Florian Pfisterer and Christoph Kern, published in the proceedings of the *ACM Conference on Fairness, Accountability, and Transparency 2024* in Rio de Janeiro, Brazil in June 2024.

You can interactively explore some of the results of the paper at [https://reliable-ai.github.io/fairml-multiverse/](https://reliable-ai.github.io/fairml-multiverse/).

## Abstract

A vast number of systems across the world use algorithmic decision making (ADM) to (partially) automate decisions that have previously been made by humans. When designed well, these systems promise more objective decisions while saving large amounts of resources and freeing up human time. However, when ADM systems are not designed well, they can lead to unfair decisions which discriminate against societal groups. The downstream effects of ADMs critically depend on the decisions made during the systems’ design and implementation, as biases in data can be mitigated or reinforced along the modeling pipeline. Many of these design decisions are made implicitly, without knowing exactly how they will influence the final system. It is therefore important to make explicit the decisions made during the design of ADM systems and understand how these decisions affect the fairness of the resulting system.
  
To study this issue, we draw on insights from the field of psychology and introduce the method of multiverse analysis for algorithmic fairness. In our proposed method, we turn implicit design decisions into explicit ones and demonstrate their fairness implications. By combining decisions, we create a grid of all possible “universes” of decision combinations. For each of these universes, we compute metrics of fairness and performance. Using the resulting dataset, one can see how and which decisions impact fairness. We demonstrate how multiverse analyses can be used to better understand variability and robustness of algorithmic fairness using an exemplary case study of predicting public health coverage of vulnerable populations for potential interventions. Our results illustrate how decisions during the design of a machine learning system can have surprising effects on its fairness and how to detect these effects using multiverse analysis.

## Running the Code

### Setup

This project uses [Pipenv](https://pipenv.pypa.io/en/latest/) to control the Python environment. To install the dependencies, first install `pipenv` on your machine, then run `pipenv sync -d` in the root directory of the project. Once set up, you can enter the virtual environment in your command line by running `pipenv shell`.

You can check whether the environment is correctly set up by activating the virtual environment (`pipenv shell`) and running the test cases via `python -m unittest`. This should output the number of tests that were run (these should be more than 0!) and a message that all tests passed (OK).

### Running the Multiverse Analysis

You can run the complete multiverse analysis by running `python multiverse_analysis.py`. Make sure to activate the virtual environment beforehand, so that the installed dependencies are available. By default this will allow you to stop and restart the analysis between different universe runs.

To explore the individual analyses conducted in each *universe* of the *multiverse*, we recommend examining `universe_analysis.ipynb`. This notebook will be executed many times with different settings for each universe.

### Analysing the Results

The different Jupyter notebooks prefixed with `analysis` are analyzing the generated output from the multiverse analysis. To compute e.g. the different measures of variable importance, you can run the notebook [`analysis_var_imp_overall.ipynb`](./analysis_var_imp_overall.ipynb). The `analysis__setup.ipynb` is used for loading and preparing the multiverse analysis results and is called by the other notebooks internally. You may wish to change this notebook, though, to choose the correct `run` to analyze.

## Examining the Generated Data

The generated data from the different analyses is located in the `output` directory. Raw data from the different *universes* can be found under `output/runs/`, raw data from the analyses e.g. the FANOVAs can be found under `output/analyses/`.

# Adapting the Analysis

We purposefully created our analysis in a way that makes it easy to adapt for your own usecase(s). The steps do so are as follows:

1. Clone (or fork) this repository to have a copy of the codebase.
2. Follow the steps in the "Setup" section to install dependencies and create a virtual environment.
3. *Optional:* Delete files and results from the original analysis. You can safely delete the `data/`, `misc/`, `interactive-analysis/` and `output/` directories.
4. Modify the [`universe_analysis.ipynb`](./universe_analysis.ipynb) notebook to contain your analysis instead. All settings / options you may wish to modify as explicit decisions in the multiverse can be configured in the `universe` object.
   - We recommend verifying that your universe analysis script works correctly by trying out a few settings and running the script manually.
5. Once you are satisfied with your universe analysis, you can update the [`multiverse_analysis.py`](./multiverse_analysis.py) script to include all available options for the decisions you created in the `universe_analysis.ipynb`.
6. Execute the multiverse analysis script by running `python multiverse_analysis.py`. Make sure you are running your analysis inside the virtual environment created in step 2.

## Important Concepts

In a multiverse analysis, we analyse the complete *multiverse* of plausible ML models. This *multiverse* is constructed by combining the plausible settings / options of multiple decisions one encounters during the design of an ML system. Each *universe* in this *multiverse*, corresponds to a unique combination of different decisions.

In our analysis we differentiate between "full" *universes*, which require refitting of the machine learnung model and *sub-universes* which can be evaluated without re-fitting the model. This distinction exists only for the sake of optimization, to save time when running the analysis. It is therefore completely optional to make use of sub-universes.

When specifying the dimensions of the *multiverse*, *sub-universes* are specified by passing in a list of lists (`[["a", "b"]]`) instead of a list (`["a", "b"]`). Each universe will receive a list of options / settings then instead of just a single option.

# Reproducible Manuscript 📜

We provide a reproducible manuscript alongside the paper, which dynamically generates all figures and numbers for the paper with the exception of purely illustrative figures, which are not based off data.

The reproducible manuscript can be found in the `paper/` directory. The paper directory contains a prepared subset of the data generated by the multiverse analysis in `paper/data/` this data is generated via the script `R/prepare-paper-data.R`.

### Setup

The reproducible manuscript is created using Quarto version 1.4.549 and R version 4.2.2. Please first install the correct versions of both Quarto and R.

To improve reproducibility, the reproducible manuscript uses `renv` to fix package versions and set up a virtual environment. Please run `renv::restore()` to install the correct version of all R packages into the vertial environment.

You *may* also need to install the Quarto ACM journal extension. To do so, navigate to `paper/` in your terminal and run `quarto add quarto-journals/acm`, you will need to confirm the installation by pressing `y` twice.

### Rendering

After completing the setup above, you can render (and execute) the reproducible manuscript by running `quarto render ./multiverse-paper.qmd` from your terminal when in the `paper/` directory. This will create a PDF file next to the manuscript itself and open it in your local internet browser.

# Container Image 📦️

To make it easier to run the code and for the sake of long term reproducibility, we provide a container image that contains all the necessary dependencies. The container image is built using [Docker](https://www.docker.com/), using it with [Podman](https://podman.io/) is most likely also possible, but not yet tested.

### Running the Analysis

To run the multiverse analysis within our prebuilt container, you can run the following command:

```bash
docker run --rm --cpus=5 -v $(pwd)/output:/app/output ghcr.io/reliable-ai/fairml-multiverse
```

Please note the cpus flag here, which may be necessary based on how powerful of a machine you use. When we first conducted the analysis on an 8 core machine we did not encounter any issues, but when running the analysis on a 32 core machine we encountered issues with a race condition leading to errors upon startup due to a [bug](https://github.com/nteract/papermill/issues/511) in the Jupyter client.

If you want to verify first whether the basic execution of the analysis works, by test running a few universes, you can run the following command. We definitely recommend doing this before running the whole analysis.

```bash
docker run --rm ghcr.io/reliable-ai/fairml-multiverse pipenv run python -m unittest
```

### Building

To build the container image, run the following command in the root directory of the project:

```bash
docker build -t fairml-multiverse .
```

To check whether the image is built correctly, you can run the following command to run the test case within the container.

```bash
docker run --rm fairml-multiverse pipenv run python -m unittest
```

To run the multiverse analysis within the container you built yourself, you can run the following command:

```bash
docker run --rm --cpus=5 -v $(pwd)/output:/app/output fairml-multiverse
```

### Replications

To ensure robustness of we reran the analysis with 5 different seeds, taken from random.org (9490635, 9617311, 7076729, 108100, 2411824). To rerun the analysis with these modified seeds we recommend the following code, where the output from each analysis will be stored in a separate output directory. You can also run these commands in parallel using the `-d` flag to have them run in the background.

```bash
docker run --rm --cpus=5 --env SEED=9490635 -v $(pwd)/output-9617311:/app/output ghcr.io/reliable-ai/fairml-multiverse
docker run --rm --cpus=5 --env SEED=9617311 -v $(pwd)/output-9617311:/app/output ghcr.io/reliable-ai/fairml-multiverse
docker run --rm --cpus=5 --env SEED=7076729 -v $(pwd)/output-7076729:/app/output ghcr.io/reliable-ai/fairml-multiverse
docker run --rm --cpus=5 --env SEED=108100 -v $(pwd)/output-108100:/app/output ghcr.io/reliable-ai/fairml-multiverse
docker run --rm --cpus=5 --env SEED=2411824 -v $(pwd)/output-2411824:/app/output ghcr.io/reliable-ai/fairml-multiverse
```

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/). Please note that the ACS PUMS data used in this work is not owned by the authors and may fall under a different license.
