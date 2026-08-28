import torch


def kl_per_state(prior_actor, target_actor, states):
    with torch.no_grad():
        prior = prior_actor.distribution(states)
    target = target_actor.distribution(states)
    return (torch.log(target.scale / prior.scale) +
            (prior.scale.square() + (prior.loc - target.loc).square()) /
            (2.0 * target.scale.square()) - 0.5).sum(-1)


def kl_loss(prior_actor, target_actor, states):
    return kl_per_state(prior_actor, target_actor, states).mean()

