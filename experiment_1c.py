import numpy as np
from PIL import Image
import cv2
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import style

from constants import SIZE, MOVE_PENALTY, PICK_UP_REWARD, DROP_OFF_REWARD
from environment import Agent, PickUpCell, DropOffCell
from helper_functions import initialize_q_table, pRandom, pExploit, get_state, perform_action, calculate_new_q, create_display_environment

WAIT_TIME = 1

style.use("ggplot")

seed = int(input("Seed: "))
random.seed(seed)
run_num = 0
def experiment_1c(LEARNING_RATE, DISCOUNT, HM_STEPS):
  q_table = initialize_q_table()
  step = 0
  all_rewards = []
  while step < HM_STEPS:
    # initial environment state
    drop_cells = [DropOffCell(0,0), DropOffCell(0,4), DropOffCell(2,2), DropOffCell(4,4)]
    pick_cells = [PickUpCell(2,4), PickUpCell(3,1)]
    agent = Agent(4, 0)

    # while ai has not reached a terminal state
    session_reward = 0
    while True:
      current_state = get_state(agent, drop_cells, pick_cells)
      
      if step < 500:
        action = pRandom()
      else:
        action = pExploit(current_state, q_table)
      
      step += 1
      reward, new_state = perform_action(agent, action, drop_cells, pick_cells)
      q_table[current_state][action] = calculate_new_q(LEARNING_RATE, DISCOUNT, reward, action, current_state, new_state, q_table)
      session_reward += reward

      # visualization
      env = create_display_environment(drop_cells, pick_cells, agent)
      img = Image.fromarray(env, "RGB")
      cv2.imshow("", np.array(img))
      if (reward == PICK_UP_REWARD or reward == DROP_OFF_REWARD):
        if cv2.waitKey(WAIT_TIME) & 0xFF == ord("q"):
          break
      else:
        if cv2.waitKey(WAIT_TIME) & 0xFF == ord("q"):
          break
      
      # all drop off locations are filled
      if (len(list(filter(lambda cell: cell.has_space() == False, drop_cells))) == len(drop_cells) or step == HM_STEPS) and run_num == 1:
        break
      elif (len(list(filter(lambda cell: cell.has_space() == False, drop_cells))) == len(drop_cells) or step == HM_STEPS) and run_num != 1:
        run_num = 1
    all_rewards.append(session_reward)

  plt.plot([i for i in range(len(all_rewards))], all_rewards)
  plt.ylabel("Reward Collected")
  plt.xlabel("Session")
  plt.show()

  print(f"max reward reached: {max(all_rewards)}")

  with open(f"qtable-experiment-1c.pickle", "wb") as f:
    pickle.dump(q_table, f)

learning_rate = float(input("Learning Rate: "))
discount_rate = float(input("Discount Rate: "))
experiment_1c(learning_rate, discount_rate, 6000)
