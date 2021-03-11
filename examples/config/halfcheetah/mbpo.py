params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'HalfCheetah',
    'task': 'v2',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'actor_train_repeat': 40,
        'critic_train_repeat': 40,
        'eval_render_mode': None,
        'eval_n_episodes': 10,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_train_slower': 1,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3, # decrease it when debug
        'sample_repeat': 1, # repeatedly propose actions on one start state
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        # 'actor_real_ratio': 1,
        'critic_same_as_actor': True,
        'target_entropy': -3,
        'max_model_t': None,
        'rollout_schedule': [20, 150, 1, 1],

        'model_log_freq': 10,
    }
}