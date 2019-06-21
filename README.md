# PFSP-WT

Ant Colony Optimization applied to the Permutation Flow-Show Problem with
Weighted Tardiness. The software provides implementation of the MMAS,
M-MMAS and PACO algorithms, as well as support for bi-objective optimization.

## How to use it

The program can be called from the root of the project folder with a command of the form:

```
python run.py <path-to-instance> --method MMAS --time 30 --local-search swap --rho 0.4 --n-ants 40
```

where "method" can take the values "MMAS", "M-MMAS" and "PACO",
"local-search" can take the values "swap", "interchange" or "insertion",
rho is the pheromone trail persistence and n-ants is the number of ants in the colony.

For more command line arguments, simply type:
```
python run.py
```

## Project structure

- Folder "pfspwt" contains the source code of the project.
- Folder "data" contains the PFSP-WT instances provided along with the project statement.
- Folder "results" contains the results for the three different algorithms mentionned in the report.
- Folder "misc" contains aggregated results, statistical tests and figures.
- Folder "report" contains the source code of the report as well as a pdf version.
- "run.py" is the entry point of the program.
- "produce-results.py" is the script that has been used to generate the content of folder "results".
- "hpo.py" is the script for hyper-optimizing the different algorithms.

## Dependencies

NumPy, Numba

The project has been developped with the following versions:

* numpy == 1.15.4
* numba == 0.41.0
