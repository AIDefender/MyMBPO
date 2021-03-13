params = {
    'type': 'MBPO',
    'universe': 'gym',
    'domain': 'Walker2d',
    'task': 'v2',

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'actor_train_repeat': 20,
        'critic_train_repeat': 20,
        'eval_render_mode': None,
        'eval_n_episodes': 10,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_train_slower': 1,
        'sample_repeat': 1, # repeatedly propose actions on one start state
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'critic_same_as_actor': True,
        'target_entropy': -3,
        'max_model_t': None,
        'rollout_schedule': [20, 150, 1, 1],

        'model_log_freq': 100,
        'model_load_dir': "/home/xuezh/ray_mbpo/Walker2d/mbpo_for_model/seed:1_2021-03-11_21-38-591qwj_t97/models",
        'model_load_index': 180000,
    }
}