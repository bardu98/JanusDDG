# JanusDDG  

Instructions for using the protein stability prediction tool presented in the paper titled  *JanusDDG: A Thermodynamics-Compliant Model for
Sequence-Based Protein Stability via Two-Fronts Multi-Head
Attention*. [ArXive](https://arxiv.org/pdf/2504.03278)

## Interpretation of Results

We used the convention where a positive $\Delta\Delta G$ indicates a stabilizing mutation, while a negative value indicates a destabilizing one.

## Prerequisites

- Conda package manager (Miniconda or Anaconda installed)

## Installation

1. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate janus_env

```



## Usage

To use this tool, you need to create a `.csv` file with the following columns (the column names are mandatory):
- **ID:** Unique ID for each Mutation;
- **Sequence:**  AA Sequence;
- **MTS:** Mutation: `<oldAA><POS><newAA>_<oldAA><POS><newAA>_...` es: `A30Y_C65G`;
- **DDG:** DDG values (optional).  


```sh
python src/main.py PATH_FILE_NAME
```


This will generate a new CSV file in the Results folder with the DDG predictions from JanusDDG:
`results/result_FILE_NAME.csv`.
