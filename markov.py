import numpy as np
from random import randrange

# Discrete time, descrete-state Markov process.

# Iterates the state, based on transitions.
# Returns the new state matrix.
def iterate_state(state, transitions):
	new_state = np.dot(state, transitions)
	return new_state

# Iterates the latest state in "states", based on state transitions,
# for the given number of steps.  Iterations are added to the list.
def iterate_steps_with_history(states, transitions, steps = 1):
	new_states = states
	for i in range(0, steps):
		# Get current state.
		current_state = new_states[len(new_states) - 1]
		# Determine new state vector.
		new_state = iterate_state(current_state, transitions)
		# Add the new state to list.
		new_states.append(new_state)
	return new_states

# Generates a new realized state.
def simulate_state(state, transitions):
	# Initialize return value.
	new_state = np.zeros_like(state)
	# Count instance rows.
	instance_count = state.shape[0]
	# Count possible states.
	state_count = state.shape[1]
	# Iterate through each instance to pick a new state.
	for instance_index in range(0, instance_count):
		# Determine current state.
		current_state_index = np.flatnonzero(state[np.s_[instance_index,:]])[0]
		# Get transition weights from current state.
		state_weights = transitions[np.s_[current_state_index,:]]
		# Choose new state, weighted by transition weights.
		new_state_index = np.random.choice(range(0, state_count), p = state_weights)
		# Update return value for that instance.
		new_state[np.s_[instance_index,new_state_index]] = 1.0
	return new_state

# Generates a new realized state through steps.
def simulate_state_steps(state, transitions, steps = 1):
	# Initialize return value.
	new_state = state
	for step_index in range(0, steps):
		new_state = simulate_state(new_state, transitions)
	return new_state

# Generates realized states through steps with history.
def simulate_state_steps_with_history(states, transitions, steps = 1):
	# Initialize return value.
	new_states = states
	for step_index in range(0, steps):
		current_state = states[len(states) - 1]
		new_state = simulate_state(current_state, transitions)
		new_states.append(new_state)
	return new_states

def generate_random_initial_state(transitions, count = 1):
	# Count the number of states.
	state_count = transitions.shape[0]
	# Determine data type of transitions.
	dtype = np.result_type(transitions)
	initial_states = np.zeros((count, state_count), dtype)
	for row_index in range(0, count):
		chosen_state = randrange(0, state_count)
		initial_states[np.s_[row_index,chosen_state]] = 1.0
	return initial_states

# Numerically determine steady state of the system from a state matrix.
def determine_steady_state(state, transitions, epsilon, max_steps = 100):
	result = None
	current_state = state
	for i in range(0, max_steps):
		# Get new state.
		new_state = iterate_state(current_state, transitions)
		# Calculate changes in state vector.
		state_change = np.subtract(current_state, new_state)
		state_change_nonzero = np.where(state_change != 0.0, state_change, np.nan)
		# Determine largest magnitude of change.
		change_max = np.nanmax(np.absolute(state_change_nonzero))

		# Determine if steady state has been reached.
		if change_max < epsilon:
			# Steady state achieved.  Exit loop.
			result = new_state
			break
		# Determine if steady state could not be achieved within step limit.
		if i == max_steps:
			# Step limit reached.  Exit loop.
			result = False
			break
		# Update result for next loop iteration.
		current_state = new_state

	return result

# Builds an identity matrix from the row count of the argument.
def get_identity_from_rows(matrix, dtype = None):
	matrix_rows = matrix.shape[0]
	if dtype is None:
		dtype = np.result_type(matrix)
	diagonal = np.identity(matrix_rows, dtype)
	return diagonal

# Looks for transience.  If any initial state does not achieve 
# the average steady state, there is transience.
def detect_transience(transitions, epsilon, max_steps = 100):
	result = None
	transition_rows = transitions.shape[0]
	initial_state = get_identity_from_rows(transitions)
	steady_state_result = determine_steady_state(initial_state, transitions, epsilon * 0.01, max_steps)
	if type(steady_state_result) is not np.ndarray:
		# Steady state could not be reached.  Transience could not be determined.
		result = None
	else:
		# Find the average probability of ending up in each state.
		state_avg = np.divide(np.nansum(steady_state_result, axis=0), transition_rows)
		# Convert averages to matrix for subtraction.
		state_avg_mat = np.broadcast_to(state_avg, (transition_rows, transition_rows))
		# Find differences from average.
		state_deviations = np.subtract(state_avg_mat, steady_state_result)
		state_deviations_nonzero = np.where(state_deviations != 0.0, state_deviations, np.nan)
		# Determine largest magnitude of deviation from average.
		state_deviation_max = np.nanmax(np.absolute(state_deviations_nonzero))

		# Determine if steady state depends on initial state.
		if state_deviation_max < epsilon:
			# All steady states are similar, regardless of initial state.
			result = False
		else:
			# State states differ, depending on initial state.
			result = True

	return result

# Detects periodicity in overall system by looking for presence of self-transitions.
def detect_periodicity(transitions):
	# Initialize return value.
	result = None
	# Get diagonal elements from transitions, indicating self-transitions.
	diagonal_elements = np.diag(transitions)
	# Determine if any are non-zero.
	largest_magnitude = np.nanmax(np.absolute(diagonal_elements))
	# Determine if periodicity is present in overall system.
	if largest_magnitude != 0:
		result = False
	else:
		result = True
	return result

# Generates the transition matrix of a simple birth-death process.
# Birth occurs at end of time step.
# Death can only occur for pre-existing entities at end of time step.
# Birth cannot occur if the process is at capacity.
def generate_simple_birth_death_transitions(p_birth, p_death, capacity):
	# Initialize return value.
	# Create matrix of every state to every state.
	transitions = np.zeros((capacity, capacity))
	# Determine common probabilities.
	p_birth_and_death = p_birth * p_death
	p_no_birth_no_death = (1 - p_birth) * (1 - p_death)
	p_net_birth = p_birth * (1 - p_death)
	p_net_death = (1 - p_birth) * p_death
	# Iterate through each state to determine weights.
	for state_index in range(0, capacity):
		# Determine weights at minimum state.
		# State can only remain unchanged or increase.
		if state_index == 0:
			# Complement of birth.
			transitions[np.s_[state_index,state_index]] = 1 - p_birth
			# Birth.
			transitions[np.s_[state_index,state_index + 1]] = p_birth
		# Determine weights at maximum state.
		elif state_index == capacity - 1:
			# Death.
			transitions[np.s_[state_index,state_index - 1]] = p_death
			# Complement of death.
			transitions[np.s_[state_index,state_index]] =  1 - p_death
		# Determine intermediate states.
		else:
			# Death but no birth.
			transitions[np.s_[state_index,state_index - 1]] = p_net_death
			# No death and no birth; birth and death.
			transitions[np.s_[state_index,state_index]] = p_no_birth_no_death + p_birth_and_death
			# Birth but no death.
			transitions[np.s_[state_index,state_index + 1]] = p_net_birth
	return transitions

# Generates the transition matrix of a one-dimensional, bounded random walk.
def generate_simple_bounded_random_walk_transitions(p_increment, max_steps):
	# Initialize return value.
	# Create matrix of every state to every state.
	transitions = np.zeros((max_steps, max_steps))
	# Determine decrement probability.
	p_decrement = 1 - p_increment
	# Iterate through each state to determine weights.
	for step_index in range(0, max_steps):
		# Determine weights at minimum state.
		# Steps can only increase.
		if step_index == 0:
			transitions[np.s_[step_index,step_index + 1]] = 1.0
		# Determine weights at maximum state.
		# Steps can only decrease.
		elif step_index == max_steps - 1:
			transitions[np.s_[step_index,step_index - 1]] = 1.0
		# Determine intermediate states.
		else:
			transitions[np.s_[step_index,step_index - 1]] = p_decrement
			transitions[np.s_[step_index,step_index + 1]] = p_increment
	return transitions


if __name__ == "__main__":

	

	# Random walk transition probabilities.
	random_walk_max_steps = 3
	p_increment = 0.5
	print("\nBounded random walk transitions with {p_increment} increment probability and {steps} maximum steps.".format(p_increment = p_increment, steps = random_walk_max_steps))
	transitions_rw = generate_simple_bounded_random_walk_transitions(p_increment, random_walk_max_steps)
	print(transitions_rw)

	# Birth-death transition probabilities.
	# p_arrival: probability of a customer arriving to the checkout line after time step.
	probability_arrival = 0.33
	# p_leave: probability of a customer leaving the checkout line after time step.
	probability_leave = 0.5
	print("\nBirth-death transition probabilities.")
	print("Birth probability:", probability_arrival)
	print("Death probability:", probability_leave)
	transitions = generate_simple_birth_death_transitions(probability_arrival, probability_leave, 11)
	print(transitions)

	# Initial state of the system.
	print("\nRandom initial state:")
	state_initial = generate_random_initial_state(transitions, 2)
	print(state_initial)
	# Creating a list from an ndarray will turn the rows into list elements.
	# The list must be initialized empty.
	states = list()
	states.append(state_initial)

	print("\nSingle iteration:")
	state_new = iterate_state(state_initial, transitions)
	print(state_new)

	# Multiple step iterations.
	iteration_steps = 10
	print("\n{steps} iterations:".format(steps = iteration_steps))
	states = iterate_steps_with_history(states, transitions, iteration_steps)
	print(states[len(states)-1])

	print("\nSimulated state:")
	sim_state = simulate_state(state_initial, transitions)
	print(sim_state)

	# Multiple step simulations.
	simulation_steps = 10
	print("\n{steps} simulation steps:".format(steps = simulation_steps))
	sim_state = simulate_state_steps(sim_state, transitions, simulation_steps)
	print(sim_state)
	sim_states = list()
	sim_states.append(sim_state)

	# From single instance.
	history_simulation_steps = 3
	print("\n{steps} simulation steps with history:".format(steps = history_simulation_steps))
	history_sim_state = generate_random_initial_state(transitions)
	history_sim_states = list()
	history_sim_states.append(history_sim_state)
	history_sim_states = simulate_state_steps_with_history(history_sim_states, transitions, history_simulation_steps)
	for s in history_sim_states: print(s) 

	print("\nSteady state:")
	steady_state = determine_steady_state(state_initial, transitions, 0.00001, 1000)
	print(steady_state)

	print("\nTransience:")
	transience_detected = detect_transience(transitions, 0.0001, 1000)
	print(transience_detected)

	print("\nPeriodicity:")
	periodicity_detected = detect_periodicity(transitions)
	print(periodicity_detected)