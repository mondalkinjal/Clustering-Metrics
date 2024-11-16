Tutorials and details for using the codes and scripts.

The enviornment used for running the python codes can be reproduced by metrics.yml file.

## For Ising Model(microstates)(in directory ising_codes)
a. Simulate the ising model. Run the bash script sbatch monte_carlo_microstates.sh which will run the python file monte_carlo_microstates.py

b. Run the bash script encoded_microstates.sh to run the python file encoded_microstates.py for getting the representation for the ising model

c. Run the bash script main_code_microstate.sh. This runs the code main_code_microstates.py which will perform the clustering (Clustering.py) and run the scoring functions(scoring.py)

d. Find out the overall scores through scores_microstate.sh which will run the code scores_overall_microstates.py
e. For DBSCAN algorithm run the bash file scores_microstates_dbscan.sh. This runs the code scores_microstates_dbscan.py which will again perform the clustering (Clustering.py) and run the scoring functions(scoringDBSCAN.py)

## For Ising Model(macrostates)(in directory ising_codes)
a. Simulate the ising model. Run the bash script sbatch monte_carlo_macrostates.sh which will run the python file monte_carlo_macrostates.py

b. Run the bash script encoded_macrostates.sh to run the python file encoded_macrostates.py for getting the representation for the ising model

c. Run the bash script main_code_macrostate.sh. This runs the code main_code_macrostates.py which will perform the clustering (Clustering.py) and run the scoring functions(scoring.py)

d. Find out the overall scores through scores_macrostate.sh which will run the code scores_overall_macrostates.py

e. For DBSCAN algorithms run the bash file scores_macrostates_dbscan.sh. This runs the code scores_macrostates_dbscan.py which will again perform the clustering (Clustering.py) and run the scoring functions(scoringDBCAN.py)

## For Protein Folding (in directory folding_codes)

The protein folding trajectories were obtained from DE Shaw Research. 

a. Find and save the properties of the protein folding trajectories using properties.sh which will run the code properties.py

b. Save the distance matrix of the folding trajectories using distance_matrix.sh which will run the code distance_matrix.py

c. Run the bash script encoded_protein.sh to run the python file encoded_protein.py for getting the representation for the protein

d. Run the bash script main_code_metrics.sh. This runs the code main_code_metrics.py which will perform the clustering (Clustering.py) and run the scoring functions(scoring.py)

e. Find out the overall scores through scores_overall_metrics.sh which will run the code scores_overall_metrics.py

e. For DBSCAN algorithm run the bash file scores_dbscan.sh which will run the code scores_dbscan.py which will again perform the clustering (Clustering.py) and run the scoring functions(scoringDBCAN.py)

## For Smlt1473 bound to Mana (in directory smlt_codes)
 
The simulations for this protein-ligand system were run by our ourselves. 

a. Run the properties.py interactively

b. Run the code representation.py which will create a representation matrix of the protein bound to the ligand.

c. Run the code main_code.sh which will run the code main_code.py which will perform the clustering (Clustering.py) and run the scoring functions(scoring.py).

d. Find out the overall scores through scores_overall.py(run interactively)

e. For DBSCAN algorithm run the bash file scores_dbscan.sh which will run the code scores_dbscan.py which will again perform the clustering (Clustering.py) and run the scoring functions(scoringDBCAN.py)

## For FKBP bound to DMSO (in directory fkbp_codes)

The simulations for this protein-ligand system were run by our ourselves. 

a. Run the properties.py interactively

b. Run the code representation.py which will create a representation matrix of the protein bound to the ligand.

c. Run the code main_code.sh which will run the code main_code.py which will perform the clustering (Clustering.py) and run the scoring functions(scoring.py).

d. Find out the overall scores through scores_overall.py(run interactively)

e. For DBSCAN algorithm run the bash file scores_dbscan.sh which will run the code scores_dbscan.py which will again perform the clustering (Clustering.py) and run the scoring functions(scoringDBCAN.py)








