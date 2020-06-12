# Zero-Shot Transfer with Multiple Source Domains

The general summary of this approach is to learn a feature extractor (aka not-quite-a-state-representation) that transfers across all $n$ source domains. Then hope that it transfers to the $n+1$th target domain. All $n+1$ domains are drawn from the same distribution.

## Invariant Policy Optimization: Towards Stronger Generalization in Reinforcement Learning

https://arxiv.org/pdf/2006.01096.pdf

**Summary**: Separates state based on agent and environment. Transfers across POMDPs with different initial environment states. Shared action and observation state, reward, transition, and observation function. Uses causality (SCMs, Interventions, etc) to identiy Parent Factors (factors that cause) of reward and ignore others. Learns a representation such that a "action predictor" (aka policy) is optimal across all domains. The only real way to do this is for the feature representation to igore domain-specific stuff. In implementation, use an average policy (average of ensembles of domain-specific policies) to update the feature extractor and each domain-specific policy. Results on LQR + distractors and barely better than PPO when finding a key, opening a door, and navigating to the goal in a gridworld.

**Thoughts**: They seem to be using a constrainted optimization but then implementing it as an unconstrained optimization. The average policy isn't _really_ the optimal across domains. They implement neural "constrained" optimization by updating the weights of the constraint more than the weights of the optimization term, which is not really constrained, but functions similarly. The results are alright. They are doing better than or the same as PPO but not working from visual observations. Would be interesting to see how this average policy method works on harder domains.

# Representation Learning in Reinforcement Learning ($RL^2$)

## Deep MDP: Learning Continuous Latent Space Models for Representation Learning

https://arxiv.org/pdf/1906.02736.pdf

**Summary**: $+1$ for the inclusion of theory and general bounds (under lipschitz smoothness). Introduces a Deep MDP = latent model of the environment. Uses a representation that bounds the error of the Q-functions for different states (with global losses). Then extend to expectations over the state-action distribution rather than the entire state space. This bounds the error in predicted Q-value based on the representation and the real Q-value. They then connect Deep MDPs to bisimilar MDPs. They note that their bounds depend on the lipschitz smoothness of the Deep MDP, but switching from Wasserstein distance to another distance (MMD, Total Variation, Energy, etc) changes the type of smoothness. Empirical results are shown on a new toy domain (DonutWorld) and Atari 2600. They minimize Deep MDP losses in addition to model-free RL losses. This does better on many Atari envrionments than their model-free baseline and other auxillary losses (reconstruction, next observation prediction, next logits prediction).

**Thoughts**: The proofs are very interesting and give (theoretical) justification to things like using transition and reward prediction to regularize model-free RL agents. However, it seems that an agent should be able to do model-based RL with a learned transition and reward function (as they have here). There are a couple reasons I can think of that necessitate the model-based RL.

1. Sparse rewards are hard? If the rewards are sparse, the reward model should predict primarily 0 reward. The model-free RL agent will make the sparse rewards more common.
2. The reward/transition models have non-zero error which causes accumulating error when planning for multiple timesteps.

# Connections to Causality

## Causality for Machine Learning

https://arxiv.org/pdf/1911.10500.pdf