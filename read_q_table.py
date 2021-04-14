import pickle

def read_q_table(pickle_file_name):
  with open(pickle_file_name, "rb") as f:
    q_table = pickle.load(f)
  return q_table

pickle_file_name = input("pickle file name: ")  # ex: "qtable-experiment-1a"
table = read_q_table(pickle_file_name + ".pickle")

f = open(pickle_file_name + '.txt', "w")
for key in table:
  key_value = str(key) + ':\t\t' + str(table[key]) + '\n'

  f.write(key_value)

f.close()
