from constants import SIZE

class Agent:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.carry = False

  def __eq__(self, target):
    return self.x == target.x and self.y == target.y

  def get_location(self):
    return (self.x, self.y)
  
  def pick_up_action(self):
    if self.carry == False:
      self.carry = True
  
  def drop_off_action(self):
    if self.carry == True:
      self.carry = False
  
  def is_carrying(self):
    return self.carry

  def move_action(self, choice):
    if choice == 0:     # WEST
      self.move(x=-1, y=0)
    elif choice == 1:   # EAST
      self.move(x=1, y=0)
    elif choice == 2:   # NORTH
      self.move(x=0, y=-1)
    elif choice == 3:   # SOUTH
      self.move(x=0, y=1)

  def move(self, x, y):
    self.x += x
    self.y += y

    if self.x < 0:
      self.x = 0
    elif self.x > SIZE-1:
      self.x = SIZE - 1
    
    if self.y < 0:
      self.y = 0
    elif self.y > SIZE - 1:
      self.y = SIZE - 1

class PickUpCell:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.blocks = 8
  
  def __eq__(self, target):
    return self.x == target.x and self.y == target.y

  def get_location(self):
    return (self.x, self.y)
  
  def picked_up(self):
    self.blocks -= 1
  
  def has_blocks(self):
    return self.blocks > 0

class DropOffCell:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.blocks = 0
  
  def __eq__(self, target):
    return self.x == target.x and self.y == target.y
  
  def get_location(self):
    return (self.x, self.y)
  
  def dropped_off(self):
    self.blocks += 1
  
  def has_space(self):
    return self.blocks < 4
