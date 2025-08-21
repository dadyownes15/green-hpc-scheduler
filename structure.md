## Structure



#### Classes and core methods       

- Workload (workload_path)
    - generates a list of job in workload.all_jobs from a workload path 
- Job 
- HPCEnv
    - init(workload_path): Create envoriment from Workload, initialize cluster, carbonIntensity
    - build_observation: returns state vector
    - step(act1,act2): takes two inputs action1 and action2 from actor, and updates the new state: Returns a done flag aswell as a observation. 
    - reset(offset | none): resets the envoriment, and loads in a new random offset
- Cluster
    - can_allocate
    - allocate
    - reset
    - release

- carbonIntensity(offset, year)
    - generate_carbon_state

- reward 
    - calculate_step_reward: Takes an observation and calculates a step reward

- baseline.py
    - Script that caculates the np hard problem on a specfic reproduciable offset, which will works as our oracle method for evaluating carbon emissions.  

- Validate(experiment_path):
    - load_model:
    - simulate_scheduling(workload_file | none, offset == 0): RUns the actual scheduling based on the trained model. 
    - compare(model_list, workload_file | none). Returns a extensive compariuson between the list of models, and the workload utilizing the baseline.py 