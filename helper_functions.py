import numpy as np
import pickle
import random

from constants import SIZE, MOVE_PENALTY, PICK_UP_REWARD, DROP_OFF_REWARD

# init q table
def initialize_q_table(saved = None):
  if saved is None:
    table = {}
    for i in range(0, SIZE):
      for j in range(0, SIZE):
        for x in [True, False]:
          for s in [True, False]:
            for t in [True, False]:
              for u in [True, False]:
                for v in [True, False]:
                  table[((i, j), x, (s, t, u, v))] = [0, 0, 0, 0]
  return table

# returns random action
def pRandom():
  return random.randint(0, 3)

# returns current optimal action
def pGreedy(state, q_table):
  index_of_maximums = []
  max_q_value = max(q_table[state])
  for i in range(len(q_table[state])):
    if q_table[state][i] == max_q_value:
      index_of_maximums.append(i)
  
  if len(index_of_maximums) > 1:
    return index_of_maximums[random.randint(0, len(index_of_maximums)-1)]
  else:
    return index_of_maximums[0]

# returns current optimal action 80% of the time (remaining 20% returns random action)
def pExploit(state, q_table):
  random_value = random.random()
  if random_value < 0.2:
    return pRandom()
  else:
    return pGreedy(state, q_table)

# returns current state of agent
def get_state(agent, drop_off_cells, pick_up_cells):
  if agent.is_carrying():
      special_space_info = [cell.has_space() for cell in drop_off_cells]
  else:
    special_space_info = [cell.has_blocks() for cell in pick_up_cells]
    special_space_info.extend([False, False])
  return (agent.get_location(), agent.is_carrying(), tuple(special_space_info))

# agent performs action
def perform_action(agent, action, drop_off_cells, pick_up_cells):
  agent.move_action(action)

  on_drop_off = list(filter(lambda cell: cell.get_location() == agent.get_location(), drop_off_cells))
  on_pick_up = list(filter(lambda cell: cell.get_location() == agent.get_location(), pick_up_cells))
  if len(on_drop_off) > 0 and agent.is_carrying() and on_drop_off[0].has_space():
    agent.drop_off_action()
    on_drop_off[0].dropped_off()
    reward = DROP_OFF_REWARD
  elif len(on_pick_up) > 0 and not agent.is_carrying() and on_pick_up[0].has_blocks():
    agent.pick_up_action()
    on_pick_up[0].picked_up()
    reward = PICK_UP_REWARD
  else:
    reward = -MOVE_PENALTY

  new_state = get_state(agent, drop_off_cells, pick_up_cells)

  return (reward, new_state)

# Q Learning new Q value
def calculate_new_q(learning_rate, discount, reward, action, current_state, new_state, q_table):
  max_future_q = max(q_table[new_state])
  current_q = q_table[current_state][action]
  return (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

def sarsa_calculate_new_q(learning_rate, discount, reward, action1, action2, current_state, new_state, q_table):
  predict = q_table[current_state][action1]
  target = reward + discount * q_table[new_state][action2]
  return q_table[current_state][action1] + learning_rate * (target - predict)

# visualization helper function
def create_display_environment(drop_off_cells, pick_up_cells, agent_cell):
  AGENT_COLOR = (0, 0, 255)
  FILL_METER_COLOR = (255, 255, 255)
  DROP_OFF_COLOR = (255, 175, 0)
  PICK_UP_COLOR = (0, 255, 0)
  BLOCK_COLOR = (100, 100, 0)
  
  img_multiplier = 100
  env = np.zeros((SIZE*img_multiplier, SIZE*img_multiplier, 3), dtype=np.uint8)
  
  for cell in drop_off_cells:
    y = cell.get_location()[0]
    x = cell.get_location()[1]
    env[y*img_multiplier:(y+1)*img_multiplier, x*img_multiplier:(x+1)*img_multiplier] = DROP_OFF_COLOR
    env[y*img_multiplier + 60:(y+1)*img_multiplier - 15, x*img_multiplier + 3:(x+1)*img_multiplier - 3] = FILL_METER_COLOR
    env[y*img_multiplier + 60:(y+1)*img_multiplier - 15, x*img_multiplier + 3:(x+1)*img_multiplier - (100 - 25*cell.blocks)] = BLOCK_COLOR
  for cell in pick_up_cells:
    y = cell.get_location()[0]
    x = cell.get_location()[1]
    env[y*img_multiplier:(y+1)*img_multiplier, x*img_multiplier:(x+1)*img_multiplier] = PICK_UP_COLOR
    env[y*img_multiplier + 60:(y+1)*img_multiplier - 15, x*img_multiplier + 3:(x+1)*img_multiplier - 3] = FILL_METER_COLOR
    env[y*img_multiplier + 60:(y+1)*img_multiplier - 15, x*img_multiplier + 3:(x+1)*img_multiplier - 4 - (12*(8 - cell.blocks))] = BLOCK_COLOR
  env[agent_cell.get_location()[0]*img_multiplier + 10:(agent_cell.get_location()[0]+1)*img_multiplier - 10, agent_cell.get_location()[1]*img_multiplier + 10:(agent_cell.get_location()[1]+1)*img_multiplier - 10] = AGENT_COLOR
  if agent_cell.is_carrying():
    env[agent_cell.get_location()[0]*img_multiplier + 20:(agent_cell.get_location()[0]+1)*img_multiplier - 20, agent_cell.get_location()[1]*img_multiplier + 20:(agent_cell.get_location()[1]+1)*img_multiplier - 20] = BLOCK_COLOR

  return env