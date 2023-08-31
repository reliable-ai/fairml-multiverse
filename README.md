# Multiverse Analysis for Algorithmic Fairness

This repository contains the source code and generated data for the preprint "Everything, Everywhere All in One Evaluation: Using Multiverse Analysis to Evaluate the Influence of Model Design Decisions on Algorithmic Fairness".

The reproducible manuscript will be made available after publication of the paper at the latest.

## Abstract

A vast number of systems across the world use algorithmic decision making (ADM) to (partially) automate decisions that have previously been made by humans. When designed well, these systems promise more objective decisions while saving large amounts of resources and freeing up human time. However, when ADM systems are not designed well, they can lead to unfair decisions which discriminate against societal groups. The downstream effects of ADMs critically depend on the decisions made during the systems’ design and implementation, as biases in data can be mitigated or reinforced along the modeling pipeline. Many of these design decisions are made implicitly, without knowing exactly how they will influence the final system. It is therefore important to make explicit the decisions made during the design of ADM systems and understand how these decisions affect the fairness of the resulting system.
  
To study this issue, we draw on insights from the field of psychology and introduce the method of multiverse analysis for algorithmic fairness. In our proposed method, we turn implicit design decisions into explicit ones and demonstrate their fairness implications. By combining decisions, we create a grid of all possible “universes” of decision combinations. For each of these universes, we compute metrics of fairness and performance. Using the resulting dataset, one can see how and which decisions impact fairness. We demonstrate how multiverse analyses can be used to better understand variability and robustness of algorithmic fairness using an exemplary case study of predicting public health coverage of vulnerable populations for potential interventions. Our results illustrate how decisions during the design of a machine learning system can have surprising effects on its fairness and how to detect these effects using multiverse analysis.

## Running the Code

This project uses [Pipenv](https://pipenv.pypa.io/en/latest/) to control the Python environment. To install the dependencies, first install `pipenv` on your machine, then run `pipenv install` in the root directory of the project.

You can run the complete multiverse analysis by running `python multiverse_analysis.py`. By default this will allow you to stop and restart the analysis between different universe runs.

To explore the individual analyses conducted in each *universe* of the *multiverse*, we recommend examining `universe_analysis.ipynb`. This notebook will be executed many times with different settings for each universe.

The different Jupyter notebooks prefixed with `analysis` are analyzing the generated output from the multiverse analysis.

## Examining the Generated Data

The generated data from the different analyses is located in the `output` directory. Raw data from the different *universes* can be found under `output/runs/`, raw data from the analyses e.g. the FANOVAs can be found under `output/analyses/`.


