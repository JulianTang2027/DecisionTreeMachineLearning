class Node:

  leaf_nodes = []

  def __init__(self, attribute_name=None, previous_feature=None, remaining_data=None, parent_node=None, label=None):
    
    self.previous_feature = previous_feature
    self.attribute_name = attribute_name
    self.children = {}
    self.remaining_data = remaining_data
    self.parent_node = parent_node
    self.label = label
  