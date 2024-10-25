

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
@misc{madakkatel2024llpowershap,
      title={LLpowershap: Logistic Loss-based Automated Shapley Values Feature Selection Method}, 
      author={Iqbal Madakkatel and Elina Hyppönen},
      year={2024},
      eprint={2401.12683},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

The manuscript can be found [here](https://arxiv.org/abs/2401.12683).

---


## License

This package is available under the MIT license. 

We acknowledge the source code made available by the authors of the package powershap available at https://github.com/predict-idlab/powershap, which were selectively modified by us to create *LLpowershap*. 

