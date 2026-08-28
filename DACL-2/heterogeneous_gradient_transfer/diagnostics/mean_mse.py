def mse_per_state(prior_actor, target_actor, states):
    prior_mean = prior_actor.distribution(states).loc.detach()
    target_mean = target_actor.distribution(states).loc
    return (prior_mean - target_mean).square().sum(-1)


def mse_loss(prior_actor, target_actor, states):
    return mse_per_state(prior_actor, target_actor, states).mean()

