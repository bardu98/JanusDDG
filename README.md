# JanusDDG  

Protein stability prediction tool presented in the *JanusDDG* paper.

## Prerequisites

- Conda package manager (Miniconda or Anaconda installed)

## Installation

1. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate janus_env

```



## Usage

To use this tool, you need to create a `.csv` file with the following columns:  

- **ID** Unique ID for each Mutation;
- **Sequence**  AA Sequence;
- **MTS** Mutation: `<oldAA><POS><newAA>_<oldAA><POS><newAA>_...` es: `A30Y_C65G`;
- **DDG** DDG values (optional).  


```sh
python src/main.py PATH_FILE_NAME
```


This will generate a new CSV file in the Results folder:
results/result_FILE_NAME.csv
