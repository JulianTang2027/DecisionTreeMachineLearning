from Node import Node
import math
import random


# File organization notes #
# ID3, Prune, Evaluate, and Test will be clearly labeled with any
# helper functions specifically for those put beneath them.
# General helper functions will be put in their own section at the bottom
# of file


""" ID3 and sub-functions """


def ID3(examples, default):
    """
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    """
    start_node = Node()

    # Empty data set handling
    if not examples:
        return default

    # Fill missing data points
    examples = fill_missing_data(examples)
    # Copy updated set
    examples_copy = [{key: value for key, value in item.items()} for item in examples]

    target_dict, num_observations = get_attribute_data(examples, "Class")
    first_target = list(target_dict.keys())[0]

    # Check if data is homogenous
    if target_dict[first_target]["Total"] == num_observations:
        # True -> Set node
        start_node.label = first_target
        start_node.remaining_data = examples
        start_node.is_leaf = True

        return start_node

    else:
        # False -> check if the attributes remaining = 0 (nothing to split on)
        if get_attributes(examples) == []:
            # True -> Check for most common target in data set
            #         this will be the 'most likely' result if you picked at random
            most_common_target = (None, 0)

            for target in target_dict:
                if target_dict[target] > most_common_target[1]:
                    most_common_target = (target, target_dict[target])

                start_node.remaining_data = []
                start_node.label = most_common_target[0]
                start_node.is_leaf = True

            return start_node

    # If we made it here that means we didn't hit a stopping point
    # need to make the tree now

    # Decide best split best off information gain
    starting_entropy = get_entropy_from_data(examples, "Class", is_first_node=True)

    atri_to_split: tuple = choose_feature_split(examples, starting_entropy)

    split_data_dict = split_data(examples, atri_to_split[0])
    start_node.remaining_data = examples
    start_node.parent_node = None
    start_node.attribute_name = atri_to_split[0]

    # For each feature in a given attribute, call 'finished_tree' recursivley
    # until all paths/branches reach a leaf node (stopping point)
    for key in split_data_dict:
        finished_tree(split_data_dict[key], key, start_node, atri_to_split[2])

    # We alter 'examples' while training, so set it back to the copy we made
    # to deal with other functions that use the same instance we use in the parameter
    examples = [{key: value for key, value in item.items()} for item in examples_copy]

    # Finished iterations, can safely return start_node which represents entire tree
    return start_node


# Recursively builds tree using a depth-first algorithm
def finished_tree(
    remaining_data, previous_feature, parent_node: Node, parent_entropy: float
) -> bool:
    # Looking for a stopping point

    # Data is pure/homogenous - first stopping criteria
    if data_is_pure(remaining_data):
        # Create new node and return a bool signifying this branch is done
        this_node = Node(None, previous_feature, remaining_data, parent_node)
        parent_node.children[previous_feature] = Node(
            None,
            previous_feature,
            remaining_data,
            parent_node,
            label=remaining_data[0]["Class"],
        )
        return True

    # Second stopping criteria: can't split data any more aka no attributes left
    else:
        # Check there are no attributes left
        if get_attributes(remaining_data) == []:

            # True -> find most common target
            most_common_target = (None, 0)
            target_dict, _ = get_attribute_data(remaining_data, "Class")
            for target in target_dict:
                if target_dict[target]["Total"] > most_common_target[1]:
                    most_common_target = (target, target_dict[target]["Total"])

            parent_node.children[previous_feature] = Node(
                None,
                previous_feature,
                remaining_data,
                parent_node,
                label=most_common_target[0],
            )
            return True

    # Neither stopping criteria met -> split data
    # 1.) Get attribute w/ highest info gain
    best_attribute = choose_feature_split(
        remaining_data, parent_entropy
    )  # -> tuple([feature_name, info_gain, entropy])

    # 2.) Split data -> removes 'best_attribute' from data set and returns a dictionary of lists of dictionaries
    # e.g. {"red": [{"points": "yes", "Size": "large", "Class"}, ...], "green": [...], ...}
    split_data_dict = split_data(remaining_data, best_attribute[0])

    # Make a new branch for each of the features and return the output
    # IMPORTANT: This is recursive

    this_node = Node(best_attribute[0], previous_feature, remaining_data, parent_node)
    parent_node.children[previous_feature] = this_node
    for key in split_data_dict:
        # repeat process
        finished_tree(split_data_dict[key], key, this_node, best_attribute[2])


def get_attribute_data(data: list, feature_name: str, target="Class"):
    """
    Returns a dictionary of dictionaries with each attribute from the feature as keys
    and the values being dictionaries with targets as keys and their counts as values along with # of observations
    E.G. "Color" feature from mushroom example -> {"red": {"Total": 2, "toxic": 1, "eatable": 1}, "green": ...}

    Parameters:
    data: list of dictionaries (what parse() outputs)
    feature_name: Name of feature to extract data of
    target: Final column with result (all data sets use "Class" as the column header for targets)
    """

    feature_dict = {}
    feature_data = []

    try:
        # For each row in data, makes tuple of that rows attribute and target
        # i.e. [("red", "eatable"), ("brown", "toxic"), ...]
        feature_data = list(
            zip([row[feature_name] for row in data], [row[target] for row in data])
        )

    except Exception as e:
        print(f"Error occured: {e}")
        return

    for cur_atrbt, cur_target in feature_data:
        atrbt_exists = cur_atrbt in feature_dict

        # Check if attribute is in dictionary yet
        if not atrbt_exists:
            # Initializing dict to hold targets and their counts for the attribute
            feature_dict[cur_atrbt] = {"Total": 1, cur_target: 1}
        else:
            target_dict = feature_dict[cur_atrbt]

            # Ensure we count values correctly with dictionary check
            if not cur_target in target_dict:
                target_dict[cur_target] = 1
                # Need to keep track of attributes total count for entropy calculation
                target_dict["Total"] += 1

            else:
                target_dict[cur_target] += 1
                target_dict["Total"] += 1

    return feature_dict, len(feature_data)


# get_entropy(...) -> Calculates entropy of data with first-node handling
#
# parameter: feature_dict -> dict of dicts with attributes as keys and dicts w/ targets as keys
# parameter: num_observations -> total # of values in data for entropy calc
# parameter: is_first_node -> used to handle (different) instructions for calculation if this is the first node
def get_entropy(feature_dict: dict, num_observations: int, is_first_node: False):

    feature_entropy = 0.0

    for attribute in feature_dict:
        target_dict = feature_dict[attribute]

        # Iterating over all attributes, so must handle current attribute entropy while at it
        attr_entropy = 0.0

        for cur_target in target_dict:
            # 'Total' is included in every target_dict, but not needed until later
            if cur_target == "Total":
                continue

            # First node entropy only uses "Class" feature (target) data, so it requires different instructions
            if is_first_node:
                target_probability = target_dict["Total"] / num_observations
                feature_entropy += target_probability * math.log2(target_probability)

            target_probability = target_dict[cur_target] / target_dict["Total"]
            attr_entropy += target_probability * math.log2(target_probability)

        # Must sum values of targets entropy before multiplying by neg. 1
        attr_entropy *= -1
        attr_probability = (feature_dict[attribute]["Total"]) / num_observations

        feature_entropy += attr_entropy * attr_probability

    # First node entropy uses different equation than entropy of entire feature, so multiply neg. 1 to correct
    if is_first_node:
        return feature_entropy * -1

    return feature_entropy


# get_entropy_from_data(...) -> Extrapolates differnet functions to more easily get entropy of feature with
#                               broader data set
# This is pretty straightforward, so I won't bother with parameters :-|
def get_entropy_from_data(data: list, attribute_name: str, is_first_node: False):

    if get_attribute_data(data, attribute_name) == None:
        filled_data = fill_missing_data(data)
        feature_dict = get_attribute_data(filled_data, attribute_name)

    feature_dict = get_attribute_data(data, attribute_name)[0]
    num_observations = get_attribute_data(data, attribute_name)[1]

    entropy = get_entropy(feature_dict, num_observations, is_first_node)

    return entropy


# get_informaiton_gain(...) -> Calculates information gain for the current node
#
# parameters: Both are super straightforward again so I'm not gonna bother :-\
def get_information_gain(parent_entropy: float, node_entropy: float) -> float:
    return parent_entropy - node_entropy


# Finds entropy of existing attributes and splits based off highest information gain
def choose_feature_split(data: list, parent_entropy: float):

    # Setting default values for first element of loop -> tuple(feature_name, info_gain, entropy)
    best_feature = ("foo", -1, None)
    for cur_feature in get_attributes(data):
        cur_entropy = get_entropy_from_data(data, cur_feature, is_first_node=False)

        cur_info_gain = get_information_gain(parent_entropy, cur_entropy)
        if cur_info_gain > best_feature[1]:
            best_feature = (cur_feature, cur_info_gain, cur_entropy)

    return best_feature


def split_data(original_data, feature_to_split):

    possible_attributes = {}

    for observation in original_data:
        # {"color": "red", "points": "yes", "Size": "Small", "Eatiablility": "Eatable"}

        cur_attr = observation[feature_to_split]  # -> "red"
        del observation[feature_to_split]

        if not cur_attr in possible_attributes:
            # Remove the feature we are splitting on
            possible_attributes[cur_attr] = [observation]
        else:
            possible_attributes[cur_attr].append(observation)

    return possible_attributes


""" End ID3 functions """


""" Start Prune and subfunctions"""


def prune(node, examples):
    """
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    """
    # rename for simplicity
    root_node = node

    # Make sure this isn't the only node - nothing to prune
    if root_node.children == {}:
        return

    # Reduces need to always calculate trees accuracy (expensive and long)
    # IMPORTANT: this only updates when pruning a sub-tree results in higher accuracy, and thus tree gets changed
    recent_tree_accuracy = test(root_node, examples)

    # Finds all leaves with distint parents w/ dict ordering: {depth: {parent_node: leaf_node}}
    # e.g. -> {4: {parent_node1: node_1}, {parent_node2: node2}, ...}
    all_leaf_nodes = get_leaf_nodes(root_node)

    # Orders dict by depth (greatest to least)
    all_leaf_nodes = dict(sorted(all_leaf_nodes.items(), reverse=True))

    # Begin recursion
    prune_nodes(all_leaf_nodes, root_node, recent_tree_accuracy, examples)


# Prune() sub-function
# Recursive function to prune nodes from bottom up
def prune_nodes(
    leaf_nodes_by_depth: dict,
    root_node: Node,
    cur_tree_accuracy: float,
    validation_data: list,
):
    # Will be used to recursivly call the function
    new_leaf_nodes = {}

    for cur_depth in leaf_nodes_by_depth:
        # Travel up tree with the current depth and test nodes at said depth
        for leaf_node in leaf_nodes_by_depth[cur_depth].values():

            if leaf_node == root_node:
                return  # nothing to consolidate if root is already a leaf node

            # Get internal node (parent)
            internal_node = leaf_node.parent_node

            if internal_node == root_node:
                majority_target = find_majority_target(internal_node.remaining_data)
                test_node = Node(
                    None,
                    internal_node.previous_feature,
                    internal_node.remaining_data,
                    None,
                    majority_target,
                )
                this_tree_accuracy = test(test_node, validation_data)

                if this_tree_accuracy > cur_tree_accuracy:
                    # If the node we are consolidating into is the root
                    # there is nothing more to prune as all children of root will be pruned in this step
                    # Maintain instance of root_node by changing its object reference (test_node)
                    root_node = test_node

                    # If consolidating into the root node was effective, there are no other prunes to do
                    return

                else:
                    # Didn't consolidate into root -> other new leaf nodes may be effective in next round
                    break

            # Get most common target from pre-partitioned data to simplify tree
            majority_target = find_majority_target(internal_node.remaining_data)

            # Make calls to parent more simple (need this for swapping subtree out)
            # IMPORTANT: If the parent is the root node, the grandparent node is None
            grandparent_node = internal_node.parent_node

            # New node (with most common target) to test if pruning was effective
            test_node = Node(
                None,
                internal_node.previous_feature,
                internal_node.remaining_data,
                grandparent_node,
                majority_target,
            )

            # Temporarily replace the branch leading from the grandparent to the test node (internal node no longer in tree)
            # Because they use the same previous_feature, we are able to do this
            if grandparent_node != None:
                grandparent_node.children[test_node.previous_feature] = test_node

            # Test accuracy using validation set
            this_tree_accuracy = test(root_node, validation_data)
            if this_tree_accuracy > cur_tree_accuracy:
                # Prune resulted in higher accuracy -> update accuracy
                cur_tree_accuracy = this_tree_accuracy

                # Check if sibling will be tested in this round of pruning OR the next round (new_leaf_nodes)
                updated_depth = cur_depth - 1
                if updated_depth in leaf_nodes_by_depth:
                    if test_node.parent_node in leaf_nodes_by_depth[updated_depth]:
                        # sibling will get checked in this round
                        continue

                # sibling isn't being checked in this round
                # check if sibling has been added to next rounds pruning set
                if updated_depth not in new_leaf_nodes:
                    # guaranteed sibling can't be in next set (would have same depths)
                    new_leaf_nodes[updated_depth] = {test_node.parent_node: test_node}

                else:
                    if test_node.parent_node not in new_leaf_nodes[updated_depth]:
                        new_leaf_nodes[updated_depth][test_node.parent_node] = test_node

            else:
                # Prune was ineffective -> revert tree
                if grandparent_node != None:
                    grandparent_node.children[test_node.previous_feature] = (
                        internal_node
                    )

    # done with testing this set of nodes, need to check the new nodes for pruning
    if new_leaf_nodes != {}:  # check for empty -> done
        prune_nodes(new_leaf_nodes, root_node, cur_tree_accuracy, validation_data)

    # Done when we get here


# From a given node in a tree, finds all leaves available from that point
# IMPORTANT: This finds ALL leaf nodes, not ones at a specific level in the tree
def get_leaf_nodes(cur_node: Node, cur_depth=0, leaf_nodes=None) -> dict:

    # Will occur on first instance - makes function call simpler with default parameter
    if leaf_nodes == None:
        leaf_nodes = {}

    if cur_node.label != None:  # means we have a leaf node -> add to list
        if (
            cur_depth in leaf_nodes
        ):  # this depth already has other nodes added to its list
            if (
                cur_node.parent_node not in leaf_nodes[cur_depth]
            ):  # making sure no two nodes have the same parent
                leaf_nodes[cur_depth][cur_node.parent_node] = cur_node

        else:
            # first time seeing this depth so guaranteed its sibling hasn't either
            leaf_nodes[cur_depth] = {cur_node.parent_node: cur_node}

    else:
        # Branch continues so iterate down it
        for key in cur_node.children:
            get_leaf_nodes(cur_node.children[key], cur_depth + 1, leaf_nodes)

    return leaf_nodes


""" End Prune functions """


""" Begin test function"""


def test(node, examples):

    examples = fill_missing_data(examples)

    result_tracker = {"correct": 0, "incorrect": 0}
    for example in examples:
        if evaluate(node, example) != example["Class"]:
            result_tracker["incorrect"] += 1
        else:
            result_tracker["correct"] += 1

    return result_tracker["correct"] / (
        result_tracker["incorrect"] + result_tracker["correct"]
    )


""" Begin evaluate function"""


def evaluate(node, example):
    """
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    """

    # simply to have a more intuitive name
    cur_node = node
    # Empty data set
    if cur_node.children == {}:
        return cur_node.label

    # Else go through tree
    while cur_node.children:
        # Everytime we reach a split in the tree, check available paths (children)
        for attribute in example:
            if attribute == "Class":  # couldn't find the attribute -> error
                continue
            if attribute == cur_node.attribute_name:
                # go to node with that attribute

                if example[attribute] not in cur_node.children:
                    return "EJ probably messed this one up"
                else:
                    cur_node = cur_node.children[example[attribute]]
                    break
                # get rid of attribute from example

        # We only get here if we never found an attribute (branch) to follow

    # If we are here, we are at the leaf node
    return cur_node.label


""" Begin foreset function """


# Creates forest of ID3 trees
def create_forest(data, num_trees):
    data_partitions = create_partitions(data, num_trees)
    all_trees = []
    for tree_index in range(num_trees):
        target_data = get_attribute_data(data_partitions[tree_index], "Class")[0]
        default_target = max(target_data, key=lambda k: target_data[k]["Total"])
        all_trees.append(ID3(data_partitions[tree_index], default_target))

    return all_trees


# Evenly distributes data into the num of trees you want
# 50 pieces of data and 5 trees -> 10 pieces of data for each tree
# uneven amounts get distributed
def create_partitions(data, num_trees):
    # Add enough data partitions as there will be trees
    data_partitions = [[] for i in range(num_trees)]
    # Iterate through data and split them across the tree they will be used on
    for index, observation in enumerate(data):
        data_partitions[index % num_trees].append(observation)

    return data_partitions


# Test an outcome on each tree of forest and use most common outcome
def get_forest_outcome(forest, example):
    outcomes_dict = {}

    for tree in forest:
        outcome = evaluate(tree, example)
        if outcome in outcomes_dict:
            outcomes_dict[outcome] += 1
        else:
            outcomes_dict[outcome] = 1

    # Find most commoon outcome
    return max(outcomes_dict, key=outcomes_dict.get)


""" End forest functions"""


""" General purpose functions """


def get_mode_for_attributes(data: list):
    mode_dict = {}
    for attribute in data[0]:
        if attribute != "Class":
            mode_dict[attribute] = {}

    for attribute in data[0].keys():
        if attribute == "Class":
            continue
        for observation in data:
            if observation[attribute] != "?":
                # Add to the counter

                if observation[attribute] in mode_dict[attribute]:
                    mode_dict[attribute][observation[attribute]] += 1
                else:
                    mode_dict[attribute][observation[attribute]] = 1

            else:
                # skip it
                continue

    final_dict = {}
    # Only get the msot common feature for each attribute
    for attribute in mode_dict:
        final_dict[attribute] = max(mode_dict[attribute], key=mode_dict[attribute].get)

    return final_dict


# Goes through every node in the tree and prints the information
# in a more visual way
def print_tree(cur_node: Node, depth=0):
    indent = " " * depth * 8
    if cur_node.parent_node is None:
        if cur_node.children == {}:
            print(f"{indent}Root: {cur_node.label}")
        else:
            print(f"{indent}Root: {cur_node.attribute_name}")
    else:
        if cur_node.label != None:  # means we have a leaf node -> get target
            print(
                f"{indent}Leaf -> {cur_node.parent_node.attribute_name} = {cur_node.previous_feature} | Target: {cur_node.label}"
            )
        else:
            # Otherwise the branch keeps going
            print(
                f"{indent}Branch -> {cur_node.parent_node.attribute_name} = {cur_node.previous_feature}  | Current Node: {cur_node.attribute_name}"
            )

    for key in cur_node.children:
        print_tree(cur_node.children[key], depth + 1)


# Function to check if data is pure
# Parameter: data ->  list of dictionaries that represent rows in the original data
# e.g. [{"Color": "red", "Points": "Yes", "Size": "Large", "Class": "Eatable"}, ...]
def data_is_pure(data):

    target_dict, num_observations = get_attribute_data(data, "Class")
    first_target = list(target_dict.keys())[0]

    # If the first target's - e.g. "eatable" - total isn't same as all
    # remaining rows, it can't be homogenous
    return target_dict[first_target]["Total"] == num_observations


def find_majority_target(data: list):
    target_dict = {}
    for observation in data:
        cur_target = observation["Class"]
        if cur_target in target_dict:
            target_dict[cur_target] += 1
        else:
            target_dict[cur_target] = 1

    return max(target_dict, key=target_dict.get)


# Fill missing data by finding the mode for each attribute
# and replacing the missing data point with it
def fill_missing_data(data: list) -> list:
    # Assumption based on given data/examples: "?" = missing data

    # dict -> og_key: {value: num_instances}

    # {"Color": {"red": 2, "green": 2, ...}}
    mode_dict = {}
    for observation in data:
        for attribute in observation:
            if attribute != "Class" and attribute not in mode_dict:
                mode_dict[attribute] = {}

    missing_data_tracker = {}
    index = 0
    for index, observation in enumerate(data):
        for attribute in observation:
            if attribute == "Class":
                continue

            if observation[attribute] != "?":
                # Add to the counter

                if observation[attribute] in mode_dict[attribute]:
                    mode_dict[attribute][observation[attribute]] += 1
                else:
                    mode_dict[attribute][observation[attribute]] = 1

            else:
                # Mark the missing position
                if attribute in missing_data_tracker:
                    missing_data_tracker[attribute].append(index)
                else:
                    missing_data_tracker[attribute] = [index]

    # First, go back to every row missing an attribute and add it
    for observation in data:
        missing_attributes = mode_dict.keys() - observation.keys()
        if missing_attributes != set():
            for attribute in missing_attributes:
                observation[attribute] = max(
                    mode_dict[attribute], key=mode_dict[attribute].get
                )

    # Now go to
    for attribute in missing_data_tracker:
        for index in missing_data_tracker[attribute]:
            data[index][attribute] = max(
                mode_dict[attribute], key=mode_dict[attribute].get
            )

    return data


def get_attributes(data):
    # Check if any features are left
    attributes = list(data[0].keys())
    attributes.remove("Class")
    # Important: This returns an empty list if "Class" is the only key left
    return attributes
