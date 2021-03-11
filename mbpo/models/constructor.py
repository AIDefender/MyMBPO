import numpy as np
import tensorflow as tf

from mbpo.models.fc import FC
from mbpo.models.bnn import BNN

def construct_model(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, 
					num_elites=5, session=None, model_dir=None, model_load_timestep=None, load_model=False):
	print('[ BNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))

	name = 'BNN' if not model_load_timestep else 'BNN_'+str(model_load_timestep)
	params = {'name': name, 'num_networks': num_networks, 'num_elites': num_elites, 
			  'sess': session, 'model_dir': model_dir, 'load_model': load_model}
	model = BNN(params)

	if not load_model:
		model.add(FC(hidden_dim, input_dim=obs_dim+act_dim, activation="swish", weight_decay=0.000025))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
		model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
		model.add(FC(obs_dim+rew_dim, weight_decay=0.0001))
	model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
	return model

def concat_samples(samples, key):
	return np.hstack((samples[key+'.observation'], samples[key+'.achieved_goal'], samples[key+'.desired_goal']))

def format_samples_for_training(samples, multi_goal=False):
	obs = concat_samples(samples, 'observations') if multi_goal else samples['observations']
	act = samples['actions']
	next_obs = concat_samples(samples, 'next_observations') if multi_goal else samples['next_observations']
	rew = samples['rewards']
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((rew, delta_obs), axis=-1)
	return inputs, outputs

def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))

if __name__ == '__main__':
	model = construct_model()
