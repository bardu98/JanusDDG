# JanusDDG  

![JanusDDG logo](./images/overwie%20janus%20unfolded.png)

Instructions for using the protein stability prediction tool presented in the paper titled  *JanusDDG: A Thermodynamics-Compliant Model for
Sequence-Based Protein Stability via Two-Fronts Multi-Head
Attention*. [ArXive](https://arxiv.org/pdf/2504.03278)

## Scope of Use

This tool is designed to predict the stability changes of proteins resulting from single or multiple mutations.

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

- **ID:** Unique identifier for each mutation. Example: `Mut_1` (you can use any unique name for each mutation).
- **Sequence:** Amino acid sequence of the full wild-type protein. Example: `SACGL...`.
- **MTS:** Mutations in the format `<oldAA><POS><newAA>_<oldAA><POS><newAA>_...`.  
  Positions start at 1 for the first amino acid in the sequence.  
  Example: `A30Y_C65G`
- **DDG:** Experimental ΔΔG values (optional). Example: `-0.5`


```sh
python src/main.py PATH_FILE_NAME
```


This will generate a new CSV file in the Results folder with the DDG predictions from JanusDDG:
`results/result_FILE_NAME.csv`.


## How to Cite

If you use this tool in your research, please cite the following paper:

```bibtex
@article{barducci2025janusddg,
  title={JanusDDG: A Thermodynamics-Compliant Model for Sequence-Based Protein Stability via Two-Fronts Multi-Head Attention},
  author={Barducci, Guido and Rossi, Ivan and Codic{\`e}, Francesco and Rollo, Cesare and Repetto, Valeria and Pancotti, Corrado and Iannibelli, Virginia and Sanavia, Tiziana and Fariselli, Piero},
  journal={arXiv preprint arXiv:2504.03278},
  year={2025}
}


