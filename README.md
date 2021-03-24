# My Model-Based Policy Optimization

I made various changes to the MBPO codebase, facilitating my research.
1. The actor and critic can use separate replay buffer, e.g. one with model-free data and the other with only model-argumented data.
2. The rollout can be vine-style, i.e. choose several actions from one given state.
3. Reproduce REDQ(https://arxiv.org/abs/2101.05982) and one can use Q ensemble together with MBPO. One can also train ensembled Q networks with different batch of data.
4. The learning of the dynamics model can be delayed.
5. One can save the checkpoint for the model network and policy network separately, and possibly reload them separately.
6. The standard deviation of the policy and ensembled Q can be evaluated and plotted.
7. Add gym reacher and fetch env.
8. The experiment name and checkpoint frequency can be set at command line.
