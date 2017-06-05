import subprocess
from pacman import *



#subprocess.check_output('python pacman.py -p ReinforceAgent -l originalClassic > resultfile.txt')

def run_episode(game_state, policy_grad, value_grad, sess):
    # unwrap parameters
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation, score, is_over = game_state
    # init var
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    for _ in xrange(NUM_EPISODES):
	    # calculate policy
	    obs_vector = np.expand_dims(observation, axis=0)
	    probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})
	    # get the index of the maximum prob action
	    action = probs.index(max(probs[0]))
	    action_dir = translateIdxToAction(action_idx)
	    #action = 0 if random.uniform(0,1) < probs[0][0] else 1
	    # record the transition
	    states.append(observation)
	    actionblank = np.zeros(POSS_ACTIONS)
	    actionblank[action] = 1
	    actions.append(actionblank)
	    # take the action in the environment
	    old_observation = observation
	    # we actually take the action





	    reward = 0

	    #observation, reward, done, info = env.step(action)
	    transitions.append((old_observation, action, reward))
	    totalreward += reward

	    if is_over:
	        break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward





if __name__ == '__main__':

    args = readCommand( sys.argv[1:] ) # Get game components based on input
    runGames( **args )
    #runGames(layout='originalClassic', pacman='MinimaxAgent', ghosts='RandomGhost', display=False,
    #		 numGames=2, record=True, numTraining=0)
    # python run.py -n 3 -p 'MinimaxAgent' -l 'originalClassic' -q
    # python run.py -n 2 -p 'ReinforceAgent' -l 'smallClassic' -t

    print args
    pass

#runGames( layout, pacman, ghosts, display, numGames, record, numTraining = 0, catchExceptions=False, timeout=30 