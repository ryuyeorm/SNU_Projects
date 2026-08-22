# implementation of inspection game. 
import random 

ACTION_LIST = [
    [1,0],
    [-1,0],
    [0,1],
    [0,-1],
    [0,0]
]

# environment
p0_pos = [0,0]
p1_pos = [0,0]
p0_def = [5,0]
p1_def = [0,5]
zone_coop = [5,5]

#return the new position
"""
1. the player is in the edge of the zone
2. the player is in the middle of the zone
"""

'''
Take position as a vector and compute the new position
If position is infeasable it takes a random action. 
'''
def return_pos(p0_action, p1_action):
    # apply action
    new_p0_pos = p0_pos + p0_action
    new_p1_pos = p1_pos + p1_action

    if all(x > 0 for x in new_p0_pos):
        temp_item = ACTION_LIST.pop(p0_action)
        feasible_action = random.choice(ACTION_LIST)
        ACTION_LIST.append(temp_item)
        p0_action
    if all(x > 0 for x in new_p1_pos):
        p1_pos = new_p1_pos



