import sys, os
import re
import json
from main import main
from typing import List
from datasets import load_dataset

import pandas as pd 
from thefuzz import process, fuzz

def calculate_metrics_for_pairs(pairs):
    # Helper function to calculate recall for one pair
    def calculate_recall(predicted_list, true_list):
        predicted_set = set(predicted_list)
        true_set = set(true_list)
        relevant_retrieved = predicted_set.intersection(true_set)
        recall = len(relevant_retrieved) / len(true_set) if len(true_set) > 0 else 0
        hit = 1 if len(relevant_retrieved) > 0 else 0
        return recall, hit

    # Calculate recall and hit for each pair
    recall_scores = []
    hit_scores = []
    for pred, true in pairs:
        recall, hit = calculate_recall(pred, true)
        recall_scores.append(recall)
        hit_scores.append(hit)

    # Calculate average recall and hit rate
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    hit_rate = sum(hit_scores) / len(hit_scores) if hit_scores else 0

    return recall_scores, average_recall, hit_rate


def movie_to_ids(movie_list: List[str]) -> List[List[int]]:
    movie_data = pd.read_csv('ml-1m/movies.dat', sep='\t', engine='python', encoding='latin-1', usecols=["Name", "MovieID"])
    movie_list = pd.DataFrame(movie_list, columns=["Name"])
    movie_list["Name"] = movie_list["Name"].apply(
        lambda x: process.extractOne(x, movie_data["Name"], scorer=fuzz.ratio)[0]
    )
    data = pd.merge(movie_data, movie_list, on="Name", how="right")
    movie_ids = data["MovieID"].tolist()

    return movie_ids


dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "5core_last_out_w_his_All_Beauty", trust_remote_code=True)
test = {
    "userid": [],
    "previous_movie_seq": [],
    "future_movie_seq": []
}
for i in range(len(dataset['test'])):
    test['userid'].append(dataset['test'][i]['user_id'])
    test['previous_movie_seq'].append(dataset['test'][i]['history'].split())
    test['future_movie_seq'].append([dataset['test'][i]['parent_asin']])


contents = []
for i in range(1):
    prompt = "User\n" + str(test["userid"][i]) + "\nhas purchased products: \n" + str(test["previous_movie_seq"][i])+", \nCan you recommend some product for him/her?"
    contents.append(prompt)

num_passed = 0
results = []
for i, content in enumerate(contents):
    predicted = main(content)
    results.append([predicted,test["future_movie_seq"][i]])

print(results)
recall_scores, average_recall, hit_rate = calculate_metrics_for_pairs(results)
print("Individual Recall Scores:", recall_scores)
print("Average Recall:", average_recall)
print("Hit Rate:", hit_rate)



