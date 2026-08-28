def wasserstein_per_state(prior_actor, target_actor, states):
    prior = prior_actor.distribution(states)
    target = target_actor.distribution(states)
    return ((prior.loc.detach() - target.loc).square() +
            (prior.scale.detach() - target.scale).square()).sum(-1)


def wasserstein_loss(prior_actor, target_actor, states):
    return wasserstein_per_state(prior_actor, target_actor, states).mean()

