import sys
from minihack.base import MiniHack
from gymnasium.utils import seeding
from nle.env.base import nethack

des_file = """
MAZE: "mylevel", ' '
FLAGS:hardfloor
INIT_MAP: solidfill,' '
GEOMETRY:center,center
MAP
--------     --------
|......|     |......|
|......|#####|......|
|......|     |......|
--------     --------
    #
    #
--------
|......|
|......|
|......|
--------
ENDMAP
REGION:(1,1,6,3),lit,"ordinary"
REGION:(14,1,19,3),lit,"ordinary"
REGION:(1,8,6,11),lit,"ordinary"

DOOR:closed,(4,4)
DOOR:closed,(7,2)
DOOR:closed,(4,7)
DOOR:closed,(13,2)

$all_rooms = selection: fillrect(1,1,6,3) & fillrect(1,8,6,10) & fillrect(14,1,19,3)
OBJECT: "full healing", rndcoord($all_rooms)
FOUNTAIN: rndcoord($all_rooms)
"""

_np_random, seed = seeding.np_random(474862)
env = MiniHack(
    des_file=des_file,
    character="@",
    actions=nethack.ACTIONS
)

seed = 474862
env.seed(core=seed, disp=seed)
obs = env.reset()
seed = _np_random.integers(sys.maxsize)
env.seed(core=seed, disp=seed)
obs = env.reset()

# This action sequence seems to work, resulting in us drinking from the fountain
working_action_sequence = [3, 7, 0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 7]
# This action sequence seems to cause an issue, resulting in us drinking from the fountain but the game still asking us if we want to drink from the fountain
# However, the popup has already been resolved pressing "y"/"n" will be treated as a normal command.
# So pressing y would be interpreted as moving north-west, while pressing n would be moving south-east
# Note how it results in "Drink from the fountain? [yn] (n) y" with an y at the end. Idk if this is intended
error_action_sequence = [3, 7, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 7]

for i, action in enumerate(error_action_sequence):
    obs, reward, done, info = env.step(action)
    waiting_for_yn = bool(env.last_observation[env._internal_index][1])
    waiting_for_line = bool(env.last_observation[env._internal_index][2])
    waiting_for_space = bool(env.last_observation[env._internal_index][3])
    print("".join([chr(c) for c in obs["message"] if c != 0]), waiting_for_yn, waiting_for_line, waiting_for_space)
