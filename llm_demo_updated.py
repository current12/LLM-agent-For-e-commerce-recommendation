from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
from math import sqrt
import re
import pandas as pd


def fetch_user_data(user_id: int) -> Dict[str, str]:
    user_data = pd.read_csv(
        'ml-1m/users.dat', sep='::', 
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding='latin-1')
    user_info = user_data.loc[user_data["UserID"] == user_id]
    
    # parse gender info
    gender = '<UKN>'
    gender_info = user_info["Gender"].values[0]
    if gender_info == 'F':
        gender = 'Female'
    elif gender_info == 'M':
        gender = 'Male'
        
    return {
        "gender": gender,
        "age": int(user_info["Age"].values[0]),
        # TODO: turn user occupation index to real occupation name
        "Occupation": int(user_info["Occupation"].values[0])
    }
    
    
    
def fetch_movie_data(movie_id: int) -> Dict[str, str]:
    movie_data = pd.read_csv('ml-1m/movies.dat', sep='\t', engine='python', encoding='latin-1')
    movie_info = movie_data[movie_data["MovieID"] == movie_id]
    
    # parse movie genres
    genres = movie_info["Genres"].values[0].split('|')
    
    return {
        "title": movie_info["Name"].values[0],
        "genres": genres
    }
    

def fetch_data(user_id: int, movie_sequence: list) -> Dict[str, Dict]:
    return {
        'user': fetch_user_data(user_id),
        'movie': [fetch_movie_data(movie_id) for movie_id in movie_sequence]
    }
    

def check_movie(movie_name: str):
    movies = pd.read_csv('movies.dat', sep="\t", engine='python', encoding='latin-1')
    if movie_name in  list(movies["Name"]):
        return True
    return False



# Do not modify the signature of the "main" function.
def main(user_query: str):

    entrypoint_agent_system_message = ""
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    entrypoint_agent = ConversableAgent("entrypoint_agent", 
                                        system_message=entrypoint_agent_system_message, 
                                        llm_config=llm_config,
                                        human_input_mode='NEVER')
    entrypoint_agent.register_for_execution(name="fetch_data")(fetch_data)
    
    fetchdata_agent_system_message = "You are an AI assistant. \
        Your role is to determine the appropriate parameters for the `fetch_data` function."
    fetchdata_agent = ConversableAgent("fetchdata_agent",
                                       system_message=fetchdata_agent_system_message,
                                       llm_config=llm_config,
                                       max_consecutive_auto_reply=1,
                                       human_input_mode='NEVER')
    fetchdata_agent.register_for_llm(name="fetch_data", description="Fetches the user information and movie information.")(fetch_data)


    analysis_agent_system_message = "Using the following analyses of a user's movie-watching preferences and demographic information, create a comprehensive profile that describes the user's likely interests, viewing habits, and personality traits. Here are the analyses:\
                                    Demographic Analysis: Based on the user’s age and gender, provide insights into the types of movies or content the user might be interested in. Consider general viewing tendencies for this demographic group and outline potential preferences in themes, styles, or types of characters the user might find appealing.\
                                    Genre Preference: Identify the genres that appear most frequently in the user's watched movies, listing genres in order of preference based on frequency. Provide insights into the top genres the user is likely interested in, using phrases like 'strong preference' or 'mild preference' to indicate the user's interest level in each genre.\
                                    Year Preference: Identify the release years that appear most frequently in the user's watched movies and provide a summary of the user's year preference. Indicate whether the user shows a strong preference for movies from certain decades (e.g., 90s, 2000s, recent releases) and describe the user’s likely interest level in each era (e.g., 'highly prefers movies from the 90s' or 'shows mild interest in recent movies').\
                                    Actor Preference: Identify any actors who appear most frequently across the user’s watched movies and provide a summary of the user’s actor preference. Rank the actors in order of preference based on frequency, and describe the user's level of interest in each (e.g., 'strong preference for movies featuring Tom Hanks' or 'shows moderate interest in Leonardo DiCaprio’s films').\
                                    Based on these findings, synthesize a cohesive user profile that reflects their overall movie-watching personality. Describe any notable trends, such as strong preferences for specific genres, decades, or actors, and how these align with the user’s demographic profile. Conclude with a summary that encapsulates the user’s unique viewing identity, highlighting any standout traits and themes that may define their movie-watching interests."
    
    rec_agent_system_message = "Using the following user preferences, recommend a list of movies that align with the user’s tastes across multiple aspects. Follow these steps to create a well-rounded set of recommendations:\
                                Genre-Based Recommendations: Recommend 20 movies that align with the user’s genre preferences. Present the output in a list.\
                                Year-Based Recommendations: Recommend 20 movies that fit the user’s preferred era or release period. Present the output in a list.\
                                Actor-Based Recommendations: Recommend 20 movies that feature the user’s preferred actors. Present the output in a list.\
                                Overall Profile-Based Recommendations: Recommend 20 movies that best align with the user’s complete profile, taking into account genre, year, actor preferences, and demographic information. Present the output in a list.\
                                After generating the lists, rank all recommended movies from highest to lowest based on their overall fit with the user's profile. Present the ranked output in a single list, starting with the highest-ranked recommendation."

    eval_agent_system_message = """
    You are simulating the user. Evaluate the recommended movies based on the user's preferences.
    For each movie, provide a response in JSON format:
    {
        "movie_title": "<title>",
        "evaluation": "Strongly Positive" | "Moderately Positive" | "Neutral" | "Negative"
    }
    Only output the list of evaluations.
    """

    check_agent_ststem_message = "check whether the recommended movies in movie.dat"
    comment_simulator_system_message = ""

    analysis_agent = ConversableAgent("analysis_agent", 
                                        system_message=analysis_agent_system_message, 
                                        llm_config=llm_config)  

    rec_agent = ConversableAgent("recommendation_agent", 
                                        system_message=rec_agent_system_message, 
                                        llm_config=llm_config) 
    
    eval_agent = ConversableAgent("evaluation_agent", 
                                        system_message=eval_agent_system_message, 
                                        llm_config=llm_config) 

    comment_simulator_agent = ConversableAgent("comment_simulator_agent", 
                                        system_message=comment_simulator_system_message, 
                                        llm_config=llm_config) 
   
    datafetch_chat_result = entrypoint_agent.initiate_chat(fetchdata_agent, message=user_query, max_turns=2)
    user_movie_info =  datafetch_chat_result.chat_history[2]['content']

    result = entrypoint_agent.initiate_chats([
        {
            "recipient": analysis_agent,
            "message": user_movie_info,
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {
            "recipient": rec_agent,
            "message": "Based on the analysis results, recommend the top 20 movies that best align with the user's preferences,only purely give the movie names(without *). If you are satisfied with the list as a high-quality match for the user's tastes, you may quit early and present the list. Ensure the output is in a python list format.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {
            "recipient": eval_agent,
            "message": "Here are the recommended movies: [Provide the list from the rec_agent's output]. Evaluate these movies from the user's perspective based on preferences, and provide detailed feedback using the structure described in the system message.",
            "max_turns": 1,
            "summary_method": "last_msg",
        },
        {
            "recipient": rec_agent,
            "message": "Based on the following evaluations, remove movies that are 'Negative' responses and replace them with new recommendations that align with the user's preferences (Must have  20 movies at last). Ensure the final list contains 20 movies ! Only provide the movie names in a Python list format and don't have other words like '#New recommendation' or '#New'.\n\nEvaluations:\n[Provide the evaluations from the eval_agent's output]",
            "max_turns": 1,
            "summary_method": "last_msg",
        }
        ])
    output = result[-1].chat_history[1]['content']


    # Use regex to find the content inside the square brackets
    match = re.search(r'\[([\s\S]+?)\]', output)

    # Extract the movie titles as a list of strings
    if match:
        list_content = match.group(1)  # Extracts the content inside the brackets
        # Split by newline, strip each line of commas, and strip surrounding quotes
        movie_list = [title.strip().strip('",') for title in list_content.splitlines() if title.strip()]
    else:
        movie_list = []
    print(movie_list)
    return movie_list
    


    
# DO NOT modify this code below.
if __name__ == "__main__":
    main(sys.argv[1])
    