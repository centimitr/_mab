def offlineEvaluate(mab, arms, rewards, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """

    history = []
    reward_history = []

    # exit when 0 rounds
    if nrounds == 0:
        return 0

    tround = 1
    idx = 0
    while tround <= nrounds:
        if idx >= len(contexts) - 1:
            break

        # get next event
        context = contexts[idx]
        arm = arms[idx]
        reward = rewards[idx]
        idx += 1

        chosen_arm = mab.play(tround, context)
        # when arm matching
        if chosen_arm == arm:
            mab.update(arm, reward, context)
            history.append((context, arm, reward))
            reward_history.append(reward)
            tround += 1

    return reward_history
