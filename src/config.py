from collections import namedtuple

TICKER = 'MSFT'

BATCH_SIZE = 128

GAMMA = 0.999

EPS_START = 0.9

EPS_END = 0.05

EPS_DECAY = 200

TARGET_UPDATE = 10

REPLAY_MEMORY_CAPACITY = 10000

NUM_EPISODES = 100

TRANSITION = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))