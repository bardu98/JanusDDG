# JanusDDG  

This repository contains the code related to the paper *JanusDDG*.  

## Usage  

To use this tool, you need to create a `.csv` file with the following columns:  

- **ID** Unique ID for each Mutation
- **Sequence**  AA Sequence
- **MTS**      Mutation: <oldAA><POS><newAA>_<oldAA><POS><newAA>_.....
- **DDG** (optional)  

Then, place the file in the `src/Data/` directory and run the following command to get predictions:  

```sh
python main.py FILE_NAME
```
This will generate a new CSV file in the Results folder:
Results/Result_FILE_NAME.csv
