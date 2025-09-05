import pulp
import os
import time
import pandas as pd


solverlist = pulp.listSolvers(onlyAvailable=True)
print(solverlist)

directory_path = "MIPLIB 2017 Part 1"

# List of all solvers to test
solvers = {
    'GLPK': pulp.GLPK_CMD(msg=False, timeLimit=3600),
    'CPLEX': pulp.CPLEX_CMD(msg=False, timeLimit=3600),
    'Gurobi': pulp.GUROBI_CMD(msg=False, timeLimit=3600),
    'Mosek': pulp.MOSEK(msg=False, timeLimit=3600),
    'PuLP CBC': pulp.PULP_CBC_CMD(msg=False, timeLimit=3600),
    'COIN': pulp.COIN_CMD(msg=False, timeLimit=3600),
    'SCIP': pulp.SCIP_PY(msg=False, timeLimit=3600),
    'HiGHS': pulp.HiGHS(msg=False, timeLimit=3600),
    'COPT': pulp.COPT_CMD(msg=False, timeLimit=3600)
}


# Initalise pandas dataframe to store the times of each solver
speed_matrix = pd.DataFrame(columns=['GLPK', 'CPLEX', 'Gurobi', 'Mosek', 'PuLP CBC', 'COIN', 'SCIP', 'HiGHS', 'COPT'])


# Initialise the solver to run through the problems
current_solver = 'GLPK'



problems = os.listdir(directory_path)

speed_matrix.insert(0, 'Problems', problems)


i = 1

# Loop through all files in the directory
for filename in problems:
    file_path = os.path.join(directory_path, filename)
    
    # Check if it's a file (and not a directory)
    if os.path.isfile(file_path):
        print(f"Processing file: {filename}")

    # Load MPS file
    try:
        # Read in the variables and the problem        
        variables, problem = pulp.LpProblem.fromMPS(file_path)
        
    except Exception as e:
        
        # Handle any read errors
        print(f"Skipping line due to an error: {e}")
        
        # Increment the problem counter
        i += 1
        
        continue
        
    
    # Start the timer
    start_time = time.time()
    
    try:
       # Solve the problem
       problem.solve(solvers[current_solver])
        
    except Exception as e:
        
        # Handle any read errors
        print(f"Skipping problem due to an error: {e}")
        
        # Increment the problem counter
        i += 1
        
        continue
    
    
    # Stop the timer
    elapsed_time = time.time() - start_time
    
    # Output the result
    print(f"Status: {pulp.LpStatus[problem.status]}")
    print(f"Objective Value: {pulp.value(problem.objective)}")
    print("Problem number: ", i)
    print("--- Solve Time {:.2f} seconds ---".format(elapsed_time))
    
    # Check whether problem was solved
    if pulp.LpStatus[problem.status] != 'Optimal':
        speed_matrix.loc[speed_matrix['Problems'] == filename, current_solver] = 'Not Solved'
    else:
        speed_matrix.loc[speed_matrix['Problems'] == filename, current_solver] = elapsed_time
    
    # Increment the problem counter
    i += 1
    
    '''
    if i >= 3:
        break
    '''


# Save the current solver times in a file
speed_matrix.to_csv("Solve Times/" + current_solver + ' ' + directory_path + '.csv', index=False)