from newID3 import (
    newID3,
    get_attribute_data,
    print_tree,
    evaluate,
    prune,
    print_feature_importance_analysis,
)
import pickle


def get_age_gap(firstage, secondage):
    try:
        age1 = int(firstage)
        age2 = int(secondage)
        gap = abs(age1 - age2)
        if gap < 4:
            return "Small"
        elif gap < 8:
            return "Medium"
        else:
            return "Large"
    except:
        return "Error"


def load_ratings():
    """Load all ratings data"""
    ratings_dict = {}
    with open("ratings.csv", "r") as f:
        next(f)
        for line in f:
            user_id, movie_id, rating, _ = line.strip().split(",")
            if movie_id not in ratings_dict:
                ratings_dict[movie_id] = []
            ratings_dict[movie_id].append((user_id, float(rating)))
    return ratings_dict


def prepare_movie_data(movie_data, ratings_dict):
    """Process movie and user data to create training examples"""
    user_gender = {}
    with open("users.csv", "r") as f:
        next(f)
        for line in f:
            user_id, gender = line.strip().split(",")[:2]
            user_gender[user_id] = gender

    training_data = []
    movie_lines = movie_data[1:]  # Skip header

    for idx, line in enumerate(movie_lines, 1):
        elem = line.split(",")
        if len(elem) < 9:
            continue

        movie_ratings = ratings_dict.get(str(idx), [])
        for user_id, rating in movie_ratings:
            if user_id in user_gender:
                try:
                    movie = {
                        "User_Gender": user_gender[user_id],
                        "Lead1_Gender": elem[1].strip(),
                        "Lead2_Gender": elem[2].strip(),
                        "Writer_Gender": elem[3].strip(),
                        "Director_Gender": elem[4].strip(),
                        "Age_Gap": get_age_gap(elem[5].strip(), elem[6].strip()),
                        "Bechdel": elem[7].strip(),
                        "MPAA": elem[8].strip(),
                        "Class": "like" if rating >= 3 else "dislike",
                    }
                    if "?" not in movie.values() and "" not in movie.values():
                        training_data.append(movie)
                except (ValueError, IndexError):
                    continue

    return training_data


def train_and_prune_movie_predictor(movie_data, ratings_dict):
    train_data = prepare_movie_data(movie_data, ratings_dict)[0:10000]
    if not train_data:
        raise ValueError("No valid training data was created")

    print("\nAnalyzing feature importance")
    feature_analysis = print_feature_importance_analysis(train_data)

    target_dict, _ = get_attribute_data(train_data, "Class")
    default_class = max(target_dict, key=lambda k: target_dict[k]["Total"])
    tree = newID3(train_data, default_class)

    with open("valid_set.csv", "r") as f:
        valid_movies = f.readlines()
    validation_data = prepare_movie_data(valid_movies, ratings_dict)

    prune(tree, validation_data)

    return tree, train_data


if __name__ == "__main__":
    print("Loading training data...")
    with open("training_set.csv", "r") as f:
        train_movies = f.readlines()

    print("Loading ratings...")
    ratings_dict = load_ratings()

    print("Training model...")
    prunetree, train_data = train_and_prune_movie_predictor(train_movies, ratings_dict)

    # Save the trained tree
    with open("pruned_movie_tree.pkl", "wb") as f:
        pickle.dump(prunetree, f)

    print("Model trained and saved to movie_tree.pkl")

    # Print training statistics
    train_likes = sum(1 for x in train_data if x["Class"] == "like")
    print(f"\nTraining Data Statistics:")
    print(f"Total examples: {len(train_data)}")
    print(f"Likes: {train_likes/len(train_data)*100:.1f}%")
