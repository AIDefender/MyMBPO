## adapted from https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py

import os
import math
import pickle
from collections import OrderedDict
from numbers import Number
from itertools import count
import gtimer as gt
import pdb

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from softlearning.algorithms.rl_algorithm import RLAlgorithm
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool

from mbpo.models.constructor import construct_model, format_samples_for_training
from mbpo.models.fake_env import FakeEnv
from mbpo.utils.writer import Writer
from mbpo.utils.visualization import visualize_policy
from mbpo.utils.logging import Progress
import mbpo.utils.filesystem as filesystem


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class MBPO(RLAlgorithm):
    """Model-Based Policy Optimization (MBPO)

    References
    ----------
        Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine. 
        When to Trust Your Model: Model-Based Policy Optimization. 
        arXiv preprint arXiv:1906.08253. 2019.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            static_fns,
            plotter=None,
            tf_summaries=False,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,

            deterministic=False,
            model_train_freq=250,
            model_train_slower=1,
            num_networks=7,
            num_elites=5,
            num_Q_elites=2, # The num of Q ensemble is set in command line
            model_retain_epochs=20,
            rollout_batch_size=100e3,
            real_ratio=0.1,
            critic_same_as_actor=True,
            rollout_schedule=[20,100,1,1],
            hidden_dim=200,
            max_model_t=None,
            dir_name=None,
            evaluate_explore_freq=0,
            num_Q_per_grp=2,
            num_Q_grp=1,
            cross_grp_diff_batch=False,

            model_load_dir=None,
            model_load_index=None,
            model_log_freq=0,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
            critic_same_as_actor ('bool'): If True, use the same sampling schema
                (model free or model based) as the actor in critic training. 
                Otherwise, use model free sampling to train critic.
        """

        super(MBPO, self).__init__(**kwargs)

        if training_environment.unwrapped.spec.id.find("Fetch") != -1:
            # Fetch env
            obs_dim = sum([i.shape[0] for i in training_environment.observation_space.spaces.values()]) 
            self.multigoal = 1
        else:
            obs_dim = np.prod(training_environment.observation_space.shape)
        # print("====", obs_dim, "========")

        act_dim = np.prod(training_environment.action_space.shape)

        # TODO: add variable scope to directly extract model parameters
        self._model_load_dir = model_load_dir
        print("============Model dir: ", self._model_load_dir)
        if model_load_index:
            latest_model_index = model_load_index
        else:
            latest_model_index = self._get_latest_index()
        self._model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim, num_networks=num_networks, num_elites=num_elites,
                                      model_dir=self._model_load_dir, model_load_timestep=latest_model_index, load_model=True if model_load_dir else False)
        self._static_fns = static_fns
        self.fake_env = FakeEnv(self._model, self._static_fns)

        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._model.name)
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        self._rollout_schedule = rollout_schedule
        self._max_model_t = max_model_t

        # self._model_pool_size = model_pool_size
        # print('[ MBPO ] Model pool size: {:.2E}'.format(self._model_pool_size))
        # self._model_pool = SimpleReplayPool(pool._observation_space, pool._action_space, self._model_pool_size)

        self._model_retain_epochs = model_retain_epochs

        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._real_ratio = real_ratio

        self._log_dir = os.getcwd()
        self._writer = Writer(self._log_dir)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_ensemble = len(Qs)
        self._Q_elites = num_Q_elites
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('[ MBPO ] Target entropy: {}'.format(self._target_entropy))

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        # self._critic_train_repeat = kwargs["critic_train_repeat"]
        # actor UTD should be n times larger or smaller than critic UTD
        assert self._actor_train_repeat % self._critic_train_repeat == 0 or \
               self._critic_train_repeat % self._actor_train_repeat == 0

        self._critic_train_freq = self._n_train_repeat // self._critic_train_repeat
        self._actor_train_freq = self._n_train_repeat // self._actor_train_repeat
        self._critic_same_as_actor = critic_same_as_actor
        self._model_train_slower = model_train_slower
        self._origin_model_train_epochs = 0

        self._dir_name = dir_name
        self._evaluate_explore_freq = evaluate_explore_freq

        # Inter-group Qs are trained with the same data; Cross-group Qs different.
        self._num_Q_per_grp = num_Q_per_grp
        self._num_Q_grp = num_Q_grp
        self._cross_grp_diff_batch = cross_grp_diff_batch

        self._model_log_freq = model_log_freq

        self._build()

    def _build(self):
        self._training_ops = {}
        self._actor_training_ops = {}
        self._critic_training_ops = {} if not self._cross_grp_diff_batch else \
                                    [{} for _ in range(self._num_Q_grp)]
        self._misc_training_ops = {} # basically no feeddict is needed
        # device = "/device:GPU:1"
        # with tf.device(device):
        #     self._init_global_step()
        #     self._init_placeholders()
        #     self._init_actor_update()
        #     self._init_critic_update()
        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()

    def _train(self):
        
        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        policy = self._policy
        pool = self._pool
        model_metrics = {}

        if not self._training_started:
            self._init_training()

            self._initial_exploration_hook(
                training_environment, self._initial_exploration_policy, pool)

        self.sampler.initialize(training_environment, policy, pool)

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        self._training_before_hook()

        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):

            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            if self._evaluate_explore_freq != 0 and self._epoch % self._evaluate_explore_freq == 0:
                self._evaluate_exploration()

            self._training_progress = Progress(self._epoch_length * self._n_train_repeat)
            start_samples = self.sampler._total_samples
            for i in count():
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                if (samples_now >= start_samples + self._epoch_length
                    and self.ready_to_train):
                    break
                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                if self._timestep % self._model_train_freq == 0 and self._real_ratio < 1.0:
                    
                    self._training_progress.pause()
                    print('[ MBPO ] log_dir: {} | ratio: {}'.format(self._log_dir, self._real_ratio))
                    print('[ MBPO ] Training model at epoch {} | freq {} | timestep {} (total: {}) | epoch train steps: {} (total: {}) | times slower: {}'.format(
                        self._epoch, self._model_train_freq, self._timestep, self._total_timestep, self._train_steps_this_epoch, self._num_train_steps, self._model_train_slower)
                    )

                    if self._origin_model_train_epochs % self._model_train_slower == 0:
                        model_train_metrics = self._train_model(batch_size=256, max_epochs=None, holdout_ratio=0.2, max_t=self._max_model_t)
                        model_metrics.update(model_train_metrics)
                        gt.stamp('epoch_train_model')
                    else:
                        print('[ MBPO ] Skipping model training due to slowed training setting')
                    self._origin_model_train_epochs += 1
                    
                    self._set_rollout_length()
                    self._reallocate_model_pool()
                    model_rollout_metrics = self._rollout_model(rollout_batch_size=self._rollout_batch_size, deterministic=self._deterministic)
                    model_metrics.update(model_rollout_metrics)
                    
                    if self._model_log_freq != 0 and self._timestep % self._model_log_freq == 0:
                        self._log_model()

                    gt.stamp('epoch_rollout_model')
                    # self._visualize_model(self._evaluation_environment, self._total_timestep)
                    self._training_progress.resume()

                self._do_sampling(timestep=self._total_timestep)
                gt.stamp('sample')

                if self.ready_to_train:
                    self._do_training_repeats(timestep=self._total_timestep)
                gt.stamp('train')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            gt.stamp('training_paths')
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment)
            gt.stamp('evaluation_paths')

            training_metrics = self._evaluate_rollouts(
                training_paths, training_environment)
            gt.stamp('training_metrics')
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}

            self._epoch_after_hook(training_paths)
            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                *(
                    (f'model/{key}', model_metrics[key])
                    for key in sorted(model_metrics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)


            yield diagnostics

        self.sampler.terminate()

        self._training_after_hook()

        self._training_progress.close()

        yield {'done': True, **diagnostics}
    
    def _evaluate_exploration(self):
        print("=============evaluate exploration=========")

        # specify data dir
        base_dir = "/home/linus/Research/mbpo/mbpo_experiment/exploration_eval"
        if not self._dir_name:
            return
        data_dir = os.path.join(base_dir, self._dir_name)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        # specify data name
        exp_name = "%d.pkl"%self._epoch
        path = os.path.join(data_dir, exp_name)

        evaluation_size = 3000
        action_repeat = 20
        batch = self.sampler.random_batch(evaluation_size)
        obs = batch['observations']
        actions_repeat = [self._policy.actions_np(obs) for _ in range(action_repeat)]

        Qs = []
        policy_std = []
        for action in actions_repeat:
            Q = []
            for (s,a) in zip(obs, action):
                s, a = np.array(s).reshape(1, -1), np.array(a).reshape(1, -1)
                Q.append(
                    self._session.run(
                        self._Q_values,
                        feed_dict = {
                            self._observations_ph: s,
                            self._actions_ph: a
                        }
                    )
                )
            Qs.append(Q)
        Qs = np.array(Qs).squeeze()
        Qs_mean_action = np.mean(Qs, axis = 0) # Compute mean across different actions of one given state.
        if self._cross_grp_diff_batch:
            inter_grp_q_stds = [np.std(Qs_mean_action[:, i * self._num_Q_per_grp:(i+1) * self._num_Q_per_grp], axis = 1) for i in range(self._num_Q_grp)]
            mean_inter_grp_q_std = np.mean(np.array(inter_grp_q_stds), axis = 0)
            min_qs_per_grp = [np.mean(Qs_mean_action[:, i * self._num_Q_per_grp:(i+1) * self._num_Q_per_grp], axis = 1) for i in range(self._num_Q_grp)]
            cross_grp_std = np.std(np.array(min_qs_per_grp), axis = 0)
        else:
            q_std = np.std(Qs_mean_action, axis=1) # In fact V std

        policy_std = [np.prod(np.exp(self._policy.policy_log_scale_model.predict(np.array(s).reshape(1,-1)))) for s in obs]

        if self._cross_grp_diff_batch:
            data = { 
                'obs': obs,
                'inter_q_std': mean_inter_grp_q_std,
                'cross_q_std': cross_grp_std,
                'pi_std': policy_std
            }
        else:
            data = {
                'obs': obs,
                'q_std': q_std,
                'pi_std': policy_std
            }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print("==========================================")


    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _log_policy(self):
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        weights = self._policy.get_weights()
        data = {'policy_weights': weights}
        full_path = os.path.join(save_path, 'policy_{}.pkl'.format(self._total_timestep))
        print('Saving policy to: {}'.format(full_path))
        pickle.dump(data, open(full_path, 'wb'))

    # TODO: use this function to save model
    def _log_model(self):
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        print('Saving model to: {}'.format(save_path))
        self._model.save(save_path, self._total_timestep)

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

    def _reallocate_model_pool(self):
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space

        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ MBPO ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._model_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
        
        elif self._model_pool._max_size != new_pool_size:
            print('[ MBPO ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            new_pool = SimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _train_model(self, **kwargs):
        env_samples = self._pool.return_all_samples()
        # train_inputs, train_outputs = format_samples_for_training(env_samples, self.multigoal)
        train_inputs, train_outputs = format_samples_for_training(env_samples)
        model_metrics = self._model.train(train_inputs, train_outputs, **kwargs)
        return model_metrics

    def _rollout_model(self, rollout_batch_size, **kwargs):
        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size
        ))

        # Keep total rollout sample complexity unchanged
        batch = self.sampler.random_batch(rollout_batch_size // self._sample_repeat)
        obs = batch['observations']
        steps_added = []
        sampled_actions = []
        for _ in range(self._sample_repeat):
            for i in range(self._rollout_length):
                # TODO: alter policy distribution in different times of sample repeating
                # self._policy: softlearning.policies.gaussian_policy.FeedforwardGaussianPolicy
                # self._policy._deterministic: False
                # print("=====================================")
                # print(self._policy._deterministic)
                # print("=====================================")
                act = self._policy.actions_np(obs)
                sampled_actions.append(act)
                
                next_obs, rew, term, info = self.fake_env.step(obs, act, **kwargs)
                steps_added.append(len(obs))

                samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term}
                self._model_pool.add_samples(samples)

                nonterm_mask = ~term.squeeze(-1)
                if nonterm_mask.sum() == 0:
                    print('[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                    break

                obs = next_obs[nonterm_mask]
        # print(sampled_actions)

        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'mean_rollout_length': mean_rollout_length}
        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length, self._n_train_repeat
        ))
        return rollout_stats

    def _visualize_model(self, env, timestep):
        ## save env state
        state = env.unwrapped.state_vector()
        qpos_dim = len(env.unwrapped.sim.data.qpos)
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]

        print('[ Visualization ] Starting | Epoch {} | Log dir: {}\n'.format(self._epoch, self._log_dir))
        visualize_policy(env, self.fake_env, self._policy, self._writer, timestep)
        print('[ Visualization ] Done')
        ## set env state
        env.unwrapped.set_state(qpos, qvel)

    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        env_batch_size = int(batch_size*self._real_ratio)
        model_batch_size = batch_size - env_batch_size

        ## can sample from the env pool even if env_batch_size == 0
        if self._cross_grp_diff_batch:
            env_batch = [self._pool.random_batch(env_batch_size) for _ in range(self._num_Q_grp)]
        else:
            env_batch = self._pool.random_batch(env_batch_size)

        if model_batch_size > 0:
            model_batch = self._model_pool.random_batch(model_batch_size)

            keys = env_batch.keys()
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch, env_batch

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })
        self._misc_training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)
        Qs_subset = np.random.choice(next_Qs_values, self._Q_elites, replace=False).tolist()
        
        # Line 8 of REDQ: min over M random indices
        min_next_Q = tf.reduce_min(Qs_subset, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        # TODO: divide it to N separate ops, where N is # of Q grps
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                    "loss", "gradients", "gradient_norm", "global_gradient_norm"
                ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})
        if self._cross_grp_diff_batch:
            assert len(Q_training_ops) >= self._num_Q_grp * self._num_Q_per_grp
            for i in range(self._num_Q_grp - 1):
                self._critic_training_ops[i].update({
                    'Q': tf.group(Q_training_ops[i * self._num_Q_grp: (i+1) * self._num_Q_grp])
                })

            self._critic_training_ops[self._num_Q_grp - 1].update({
                'Q': tf.group(Q_training_ops[(self._num_Q_grp - 1) * self._num_Q_grp:])
            })
        else:
            self._critic_training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })
            self._actor_training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        assert len(Q_log_targets) == self._Q_ensemble

        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)
        mean_Q_log_target = tf.reduce_mean(Q_log_targets, axis=0)
        Q_target = min_Q_log_target if self._Q_ensemble == 2 else mean_Q_log_target

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - Q_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})
        self._actor_training_ops.update({'policy_train_op': policy_train_op})

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        mix_batch, mf_batch = batch

        self._training_progress.update()
        self._training_progress.set_description()

        if self._cross_grp_diff_batch:
            assert len(mix_batch) == self._num_Q_grp
            if self._real_ratio != 1:
                assert 0, "Currently different batch is not supported in MBPO"
            mix_feed_dict = [self._get_feed_dict(iteration, i) for i in mix_batch]
            single_mix_feed_dict = mix_feed_dict[0]
        else:
            mix_feed_dict = self._get_feed_dict(iteration, mix_batch)
            single_mix_feed_dict = mix_feed_dict


        if self._critic_same_as_actor:
            critic_feed_dict = mix_feed_dict
        else:
            critic_feed_dict = self._get_feed_dict(iteration, mf_batch)

        self._session.run(self._misc_training_ops, single_mix_feed_dict)

        if iteration % self._actor_train_freq == 0:
            self._session.run(self._actor_training_ops, single_mix_feed_dict)
        if iteration % self._critic_train_freq == 0:
            if self._cross_grp_diff_batch:
                assert len(self._critic_training_ops) == len(critic_feed_dict)
                [
                    self._session.run(op, feed_dict)
                    for (op, feed_dict) in zip(self._critic_training_ops, critic_feed_dict)
                ]
            else:
                self._session.run(self._critic_training_ops, critic_feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """
        mix_batch, _ = batch
        if self._cross_grp_diff_batch:
            mix_batch = mix_batch[0]
        mix_feed_dict = self._get_feed_dict(iteration, mix_batch)

        # (Q_values, Q_losses, alpha, global_step) = self._session.run(
        #     (self._Q_values,
        #      self._Q_losses,
        #      self._alpha,
        #      self.global_step),
        #     feed_dict)
        Q_values, Q_losses = self._session.run(
            [self._Q_values, self._Q_losses],
            mix_feed_dict
        )

        alpha, global_step = self._session.run(
            [self._alpha, self.global_step],
            mix_feed_dict
        )

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
        })

        policy_diagnostics = self._policy.get_diagnostics(
            mix_batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables

    def _get_latest_index(self):
        if self._model_load_dir is None:
            return
        return max([int(i.split("_")[1].split(".")[0]) for i in os.listdir(self._model_load_dir)])