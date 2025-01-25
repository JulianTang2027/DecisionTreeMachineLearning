from newID3 import evaluate
import pickle
from trainfinalproject import load_ratings, prepare_movie_data


def evaluate_accuracy(tree, data):
    """Calculate accuracy on a dataset"""
    correct = 0
    total = len(data)

    for example in data:
        prediction = evaluate(tree, example)
        if prediction == example["Class"]:
            correct += 1

    return correct / total if total > 0 else 0


# Load the trained model
with open("movie_tree.pkl", "rb") as f:
    tree = pickle.load(f)

with open("pruned_movie_tree.pkl", "rb") as f:
    prunetree = pickle.load(f)

# Load ratings once since they're used for both sets
ratings_dict = load_ratings()

# Evaluate validation set
with open("valid_set.csv", "r") as f:
    valid_movies = f.readlines()
valid_data = prepare_movie_data(valid_movies, ratings_dict)
valid_accuracy = evaluate_accuracy(tree, valid_data)
prune_valid_accuracy = evaluate_accuracy(prunetree, valid_data)
print(f"Validation Accuracy: {valid_accuracy:.4f}")
print(f"Prune Validation Accuracy: {prune_valid_accuracy:.4f}")


# Evaluate test set
with open("test_set.csv", "r") as f:
    test_movies = f.readlines()
test_data = prepare_movie_data(test_movies, ratings_dict)
test_accuracy = evaluate_accuracy(tree, test_data)
prune_test_accuracy = evaluate_accuracy(prunetree, test_data)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Prune Test Accuracy: {prune_test_accuracy:.4f}")
