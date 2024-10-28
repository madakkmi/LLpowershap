

> *LLpowershap* is a **wrapper feature selection method** that uses logistic loss-based **Shapley values** and statistical methods to select relevant features with minimal noise in the output.   

## Installation

* Install the *powershap* package available at https://pypi.org/project/powershap/ (tested on v0.0.11).  More details on the package is available at https://github.com/predict-idlab/powershap.
* Replace the files, **powershap.py**, **utils.py** and **shap_explainer.py** in the installation directory of *powershap* by the files provided by us [here](source_files)

Please note that these installation files are provided for those who wish to reproduce the results. They are not intended to replace *powershap*. 

*LLpowershap* is designed to work exclusively with classification problems and specifically with the XGBClassifier.

## Usage

```py
from powershap import PowerShap
from xgboost import XGBClassifier

X, y = ...            # Get your features into X and labels into y

selector = PowerShap(
    model=XGBClassifier(n_estimators=250, early_stopping_rounds=25, verbosity=0),
    method='LLpowershap'
)

selector.fit(X, y)     # Fit the feature selector
selector.transform(X)  # Get the dataset with only the selected features

```


## Benchmarks ⏱

Check out our benchmark results [here](tests/results/).  


## Referencing

Please cite our work as:

```bibtex
@article{madakkatel2024llpowershap,
  title={LLpowershap: logistic loss-based automated Shapley values feature selection method},
  author={Madakkatel, Iqbal and Hypp{\"o}nen, Elina},
  journal={BMC Medical Research Methodology},
  volume={24},
  number={1},
  pages={247},
  year={2024},
  publisher={Springer}
}
```

The manuscript can be found [here](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-024-02370-8).

---


## License

This package is available under the MIT license. 

We acknowledge the source code made available by the authors of the package powershap available at https://github.com/predict-idlab/powershap, which were selectively modified by us to create *LLpowershap*. 

