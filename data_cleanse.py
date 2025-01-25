og_data = ["users.dat", "movies.dat", "ratings.dat"]
csv_data = ["users.csv", "movies.csv", "ratings.csv"]
import math
import random

# IMPORTANT: movies.dat can't be converted to a csv because
#            many of the titles contain commas which results in 
#            misalignment when reformatting
def convert_to_csv(og_data_list, csv_data_list, delimitter="::"):
    """
    Converts each file in the old data to a csv given the delimitter

    arg1: og_data_list -> list of file names of the original data
    arg2: csv_data_list -> list of file names for og data to be put into after csv conversion
    arg3 (default parameter): deilimitter -> text/string to split line on
    """
    for og_file, csv_file in zip(og_data_list, csv_data_list):
        read = open(og_file, "r")
        write = open(csv_file, "w")
        for line in read.readlines():
            values = line.split(delimitter)
            csv_line = ""
            for value in values:
                csv_line += (value + ",") if value != values[-1] else value
            write.write(csv_line)

def make_romance_only_file(movies_dat, romance_dat):
    """
    Takes a file of movies with various genres and copies all lines where Romance is one of the genres
    """
    all_movies = open(movies_dat, "r")
    romance_only = open(romance_dat, "w")
    for line in all_movies.readlines():
        if "Romance" in line or "romance" in line:
            romance_only.write(line)

def make_romance_only_file(romance_movies_file, file_to_clean, new_file, movie_id_index):
    """
    Finds the movie id's that are associated with Romance and 
    adjusts other files to only include rows that have a romance
    movie in them
    """
    
    romance_movie_data = open(romance_movies_file, "r")
    all_lines = romance_movie_data.readlines()
    romance_movie_ids = [cur_movie_data.split("::")[0] for cur_movie_data in all_lines]

    # All files should have first row be column headers - descriptions - so ignore it
    old_data = open(file_to_clean, "r")
    new_data = open(new_file, "w")    
    
    
    for line in old_data.readlines():
        if line.split(",")[movie_id_index] in romance_movie_ids:
            new_data.write(line)
    

def remove_quotes_from_csv(file_to_clean, new_file):
    file_to_clean = open(file_to_clean, "r", encoding="latin-1")
    new_file = open(new_file, "w")
    
    for line in file_to_clean.readlines():
        if "Romance" in line.split(",")[-1]:
            new_file.write(line)
    

def add_movie_id(romance_movies_only, romance_features, new_file):
    romance_movies_only = open(romance_movies_only, "r")
    
    
    movie_dict = {}
    
    for line in romance_movies_only.readlines()[1:]:
        features = line.split(",")
        movie_dict[features[1]] = features[0]
        
    
    romance_features = open(romance_features, "r")
    new_file = open(new_file, "w")

    for line in romance_features.readlines()[1:]:
        features = line.split(",")
        
        if features[0] in movie_dict:
            new_file.write(movie_dict[features[0]] + "," + line)


def get_romance_ratings(ratings_file, romance_movies, new_file):
    romance_movies = open(romance_movies, "r")
    
    valid_ids = []
    
    for line in romance_movies.readlines()[1:]:
        valid_ids.append(line.split(",")[0])

    new_file = open(new_file, "w")
    ratings_file = open(ratings_file, "r")
    
    for line in ratings_file.readlines()[1:]:
        movie_id = line.split(",")[1]
        
        if movie_id in valid_ids:
            new_file.write(line)
            

def splice_ratings_and_movies(valid_ratings, movie_features, user_data, splice_file):
    movie_feature_dict = {} # movieid: movie_data
    movie_features = open(movie_features, "r")
    for line in movie_features.readlines()[1:]:
        features = line.split(",")
        movie_feature_dict[features[0]] = line

    valid_ratings = open(valid_ratings, "r")
    user_data = open(user_data, "r")
    user_dict = {}
    
    for line in user_data.readlines()[1:]:
        features = line.split(",")
        user_dict[features[0]] = features[0] + ',' + features[1] + ',' + features[2] + ',' + features[3]
    
    
    splice_file = open(splice_file, "w")
    splice_file.write("User ID, User Gender, User Age, User Occupation,Movie ID, Movie Title, Lead 1 Gender, Lead 2 Gender, Writer Gender, Director Gender, Lead 1 Age, Lead 2 Age, Passes Bechdel Test, MPAA Rating, Rating\n")
    for line in valid_ratings.readlines()[1:]:
        rating_features = line.split(",")
        user_id = rating_features[0]
        movie_id = rating_features[1]
        rating = rating_features[2]
        
        if user_id not in user_dict or  movie_id not in movie_feature_dict:
            continue
        
        user_data = user_dict[user_id]
        movie_data = movie_feature_dict[movie_id]
        
        observation = user_data.replace("\n", "")  + "," + movie_data.replace("\n", "") + "," + rating.replace("\n", "")
        
        splice_file.write(observation + '\n')
        
        
def remove_ids_from_observations(observation_w_ids, new_file):
    observation_w_ids = open(observation_w_ids, "r")
    new_file = open(new_file, "w")
    
    for line in observation_w_ids.readlines()[1:]:
        observation_features = line.split(",")
        clean_observation = ""
        
        for index, feature in enumerate(observation_features):
            if index == 0 or index == 4 or index == 5:
                continue
            else:
                clean_observation += feature.replace(" ", "") + "," if index != 14 else feature.replace(" ", "")
        
        new_file.write(clean_observation)

def get_feature_possible_values(observs_no_ids):
    gender_dict = {}
    mpaa_rating_dict = {}
    
    observs_no_ids = open(observs_no_ids, "r")
    
    for line in observs_no_ids.readlines()[1:]:
        for index, feature in enumerate(line.split(",")):
            if index in [4]:
                if feature not in gender_dict:
                    gender_dict[feature] = len(list(gender_dict.values()))

            if index == 5:
                if feature not in mpaa_rating_dict:
                    mpaa_rating_dict[feature] = len(list(mpaa_rating_dict.values()))
                    
    
    print(gender_dict)
    print("\n\n")
    print(mpaa_rating_dict)
    
    
def convert_features_to_nums(observs_no_ids, new_file):
    gender_dict = {"F": "0", "Female":"0", "M": "1", "Male": "1"}
    yes_no_dict = {"No": "0", "Yes": "1"}
    MPAA_dict = {"G": "0", "PG": "1", "PG-13": "2", "R": "3", "NC-17": "4", "NotRated": "5", "Approved": "6", "Passed": "7"}
    
    observs_no_ids = open(observs_no_ids, "r")
    new_file = new_file = open(new_file, "w")
    
    for line in observs_no_ids.readlines()[1:]:
        converted_row = ""
        is_valid_row = True
        
        for idx, feature in enumerate(line.split(",")):
            if idx in [0, 3, 4, 5, 6]:
                if feature in gender_dict:
                    converted_row += gender_dict[feature] + ","
                else:
                    is_valid_row = False
                    
            elif idx == 10:
                if feature in MPAA_dict:
                    converted_row += MPAA_dict[feature] + ","
                else:
                    is_valid_row = False
            elif idx == 9:
                if feature in yes_no_dict:
                    converted_row += yes_no_dict[feature] + ","
                else:
                    is_valid_row = False

            else:
                converted_row += feature + "," if idx != 11 else feature

        # Done with converting necessary columns
        # If feature was valid value, write it into new file
        if is_valid_row:
            new_file.write(converted_row)
            
def get_age_dif(observs_converted_nums, new_file):
    observs_converted_nums = open(observs_converted_nums, "r")
    new_file = open(new_file, "w")
    for line in observs_converted_nums.readlines()[1:]:
        features = line.split(",")
        actor_one_age = int(features[7])
        actor_two_age = int(features[8])
        
        age_dif = str(abs(actor_one_age - actor_two_age))
        new_line = ""
        for idx, feature in enumerate(features):
            if idx == 7:
                new_line += age_dif + ","
            elif idx == 8:
                continue
            else:
                new_line += feature + "," if idx != 11 else feature

        new_file.write(new_line)


def correct_num_features(old_file, new_file):
    old_file = open(old_file, "r")
    new_file = open(new_file, "w")
    
    for line in old_file.readlines()[1:]:
        if len(line.split(",")) != 11:
            continue
        else:
            new_file.write(line)

def split_into_training_valid_test(all_observations, training_file, valid_file, test_file):
    all_observations = open(all_observations, "r").readlines()[1:]
    training_file = open(training_file, "w")
    valid_file = open(valid_file, 'w')
    test_file = open(test_file, "w")
    
    total_num_observs = len(all_observations)
    
    num_training_obs = int(total_num_observs * .8)
    num_valid_obs = int(total_num_observs * .1)
    num_test_obs = total_num_observs - num_training_obs - num_valid_obs
    
    training_counter = 0
    valid_counter = 0
    test_counter = 0
    
    usable_indices = [index for index in range(0, total_num_observs)]
    
    for counter in range (0, total_num_observs):
        random_index = random.choice(usable_indices)
        usable_indices.remove(random_index)
        random_observation = all_observations[random_index]
        
        if training_counter != num_training_obs:
            training_counter += 1
            training_file.write(random_observation)
            
        elif valid_counter != num_valid_obs:
            valid_counter += 1
            valid_file.write(random_observation)
            
        elif test_counter != num_test_obs:
            test_counter += 1
            test_file.write(random_observation)
            
        else:
            print("This was an extra observation")
    
        print(counter)
    
    print(usable_indices)
