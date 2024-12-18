from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import re
import json
import pandas as pd
from autogen import Cache
seed_num = 123

def fetch_user_data(user_id: int) -> Dict[str, str]:
    user_data = pd.read_csv(
        'ml-1m/users.dat', sep='::',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding='latin-1')
    user_info = user_data.loc[user_data["UserID"] == user_id]


    gender_info = user_info["Gender"].values[0]
    gender = 'Female' if gender_info == 'F' else 'Male' if gender_info == 'M' else '<UKN>'

    return {
        "gender": gender,
        "age": int(user_info["Age"].values[0]),
        "Occupation": int(user_info["Occupation"].values[0])
    }

def fetch_movie_data(movie_id: int) -> Dict[str, str]:
    movie_data = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python', encoding='latin-1', names=["MovieID", "Name", "Genres"])
    movie_info = movie_data[movie_data["MovieID"] == movie_id]


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


def main(user_query: str):

    entrypoint_agent_system_message = """
    """
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    entrypoint_agent = ConversableAgent("entrypoint_agent",
                                        system_message=entrypoint_agent_system_message,
                                        llm_config=llm_config,
                                        human_input_mode='NEVER')
    entrypoint_agent.register_for_execution(name="fetch_data")(fetch_data)

    fetchdata_agent_system_message = "You are an AI assistant. Your role is to determine the appropriate parameters for the `fetch_data` function."
    fetchdata_agent = ConversableAgent("fetchdata_agent",
                                       system_message=fetchdata_agent_system_message,
                                       llm_config=llm_config,
                                       max_consecutive_auto_reply=1,
                                       human_input_mode='NEVER')
    fetchdata_agent.register_for_llm(name="fetch_data", description="Fetches the user information and movie information.")(fetch_data)



    analysis_agent_system_message = """
    You are an AI assistant tasked with analyzing a user's movie-watching preferences and demographic information. Output the analysis in JSON format, including the following fields:
    {
        "Demographic Analysis": "<Your analysis here>",
        "Genre Preference": "<Your analysis here>",
        "Year Preference": "<Your analysis here>",
        "Summary": "<Summary of the User's Movie-Watching Personality>"
    }
    Only output the JSON object.
    """

    rec_agent_system_message = """
    Using the user's preferences, recommend a list of movies that align with the user's tastes across multiple aspects. Follow these steps to create a well-rounded set of recommendations:

    1. **For the first iteration**:
    - Generate a ranked list of 20 movies that align with the user's preferences across the following aspects:
        - **Genre-Based Recommendations**: Movies that match the user's genre preferences.
        - **Year-Based Recommendations**: Movies from the user's preferred era or release period.
        - **Actor-Based Recommendations**: Movies featuring the user's preferred actors.
        - **Overall Profile-Based Recommendations**: Movies that best align with the user's complete profile.
    - Present the list in the following JSON format:
        {
        "recommended_movies": [
            "Movie Title 1",
            "Movie Title 2",
            ...
        ],
        "movie_new": []
        }


    2. **For subsequent iterations**:
    - Maintain a list of 20 movies by:
        - Retaining movies that have already been saved as `movies_to_save`.
        - Replacing movies specified in `movies_to_remove` with new recommendations.
    - New recommendations should align with the user's preferences based on the same aspects as described above.
    - Output the updated list of 20 movies and clearly specify the newly added movies as `movie_new` (i.e., movies that replaced those in `movies_to_remove`).
    - Ensure the output is formatted as a valid JSON object:
        {
        "recommended_movies": [
            "Movie Title 1",
            "Movie Title 2",
            ...
        ],
        "movie_new": [
            "New Movie Title 1",
            "New Movie Title 2",
            ...
        ]
        }

    3. **Important Instructions**:
    - For the first iteration, `movie_new` should be an empty list as no movies are replaced.
    - For subsequent iterations, ensure that `movie_new` includes only the new movies added in the current iteration.
    - Always return the result strictly in the specified JSON format.
    """


    comment_simulator_system_message = """
    Based on the user's analysis results and the list of recommended movies, generate comments for each movie that reflect the user's likely thoughts and feelings about the movie. The comments should be in the user's voice and consider their preferences and personality traits.

    **When writing comments:**

    - Be honest and critical. If the user is likely to have negative or neutral feelings about certain aspects of the movie, reflect that in the comments.
    - Avoid giving unwarranted praise. Balance positive remarks with constructive criticism.
    - Consider the user's preferences and how each movie may or may not align with them.

    **Write comments from the following perspectives:**

    - **Plot and Storyline**: Discuss strengths and weaknesses. Was it engaging or predictable?
    - **Characters and Acting**: Evaluate performances. Were they convincing or lacking depth?
    - **Visual Effects and Cinematography**: Comment on visual appeal. Were the effects impressive or subpar?
    - **Themes and Messages**: Analyze underlying themes. Did they resonate or feel forced?
    - **Personal Impact and Enjoyment**: Reflect on overall enjoyment. Would the user watch it again or recommend it?

    **Example of a critical comment:**
    {
        "movie_title": "Example Movie",
        "comments": {
            "Plot and Storyline": "The plot was somewhat predictable and didn't offer any surprises.",
            "Characters and Acting": "Some characters felt one-dimensional, and the acting was inconsistent.",
            "Visual Effects and Cinematography": "While there were a few stunning scenes, overall the visuals were average.",
            "Themes and Messages": "The themes were interesting but not explored in depth.",
            "Personal Impact and Enjoyment": "I didn't feel particularly engaged and wouldn't be eager to rewatch it."
        }
    }
  

    Provide the comments in the following JSON format, one per movie:
    {
        "movie_title": "<title>",
        "comments": {
            "Plot and Storyline": "<comment>",
            "Characters and Acting": "<comment>",
            "Visual Effects and Cinematography": "<comment>",
            "Themes and Messages": "<comment>",
            "Personal Impact and Enjoyment": "<comment>"
        }
    }

    Only output the list of comments in JSON format.
    """

    eval_agent_system_message = """
    You are simulating the user. Evaluate the recommended movies based on the user's preferences and the comments provided.

    **Scoring Instructions:**

    - For each movie, start with a base score of 0.
    - Add 1 point for each positive comment in the following categories:
    - **Plot and Storyline**
    - **Characters and Acting**
    - **Visual Effects and Cinematography**
    - **Themes and Messages**
    - **Personal Impact and Enjoyment**
    - Do not add points for neutral comments.
    - Subtract 1 point for each negative comment (minimum total score is 0).
    - Be strict and critical. Do not give high scores unless justified by the comments.

    **Definition of Comments:**

    - **Positive Comment**: Expresses satisfaction, enjoyment, or appreciation.
    - **Neutral Comment**: Neither positive nor negative; shows indifference.
    - **Negative Comment**: Expresses disappointment, criticism, or dislike.

    **Example Evaluation:**

    If a movie has a comment structure like this:
    {
        "movie_title": "Example Movie",
        "comments": {
            "Plot and Storyline": "The storyline started strong but lost momentum halfway through. Some plot points felt underdeveloped.",
            "Characters and Acting": "The lead actor is OK, but supporting characters lacked depth.",
            "Visual Effects and Cinematography": "The cinematography was stunning, especially the scenes shot at sunset.",
            "Themes and Messages": "The movie touched on important themes of redemption, but didn't explore them fully.",
            "Personal Impact and Enjoyment": "Overall, I enjoyed parts of the movie but felt it didn't live up to its potential. I might not watch it again."
        }
    }

    The score would be:0 (base score) - 1 (Plot) + 0 (Characters) + 1 (Visual Effects) + 0 (Themes) + 0 (Personal Feelings) = 0

    **Instructions:**

    - **Output only the list of evaluations in valid JSON format.**
    - **Do not include any additional text or explanations.**
    - **Ensure the JSON is properly formatted and can be parsed by `json.loads()`.**

    Provide the evaluations in the following JSON format:
    [
        {
            "movie_title": "<title>",
            "evaluation": <score>
        },
        ...
    ]
    """

    judge_agent_system_message = """
    You are the judge agent. Based on the evaluations provided, remove movies that are rated less than 4. Provide a list of movies to be removed. If all movies are rated 4 or 5, indicate that the process is complete.

    Output your response in the following JSON format:

    {
        "movies_to_remove": [ "<movie_title1>", "<movie_title2>", ... ],
        "process_complete": true/false
    }

    Only output the JSON object.
    """


    analysis_agent = ConversableAgent("analysis_agent",
                                      system_message=analysis_agent_system_message,
                                      llm_config=llm_config)

    rec_agent = ConversableAgent("recommendation_agent",
                                 system_message=rec_agent_system_message,
                                 llm_config=llm_config)

    comment_simulator_agent = ConversableAgent("comment_simulator_agent",
                                               system_message=comment_simulator_system_message,
                                               llm_config=llm_config)

    eval_agent = ConversableAgent("evaluation_agent",
                                  system_message=eval_agent_system_message,
                                  llm_config=llm_config)
    judge_agent = ConversableAgent("judge_agent",
                                   system_message=judge_agent_system_message,
                                   llm_config=llm_config)


    datafetch_chat_result = entrypoint_agent.initiate_chat(fetchdata_agent, message=user_query, max_turns=2)
    user_movie_info = datafetch_chat_result.chat_history[2]['content']


    result = entrypoint_agent.initiate_chats([
        {
            "recipient": analysis_agent,
            "message": user_movie_info,
            "max_turns": 1,
            "summary_method": "last_msg",
        }
    ])
    analysis_output = result[-1].chat_history[1]['content']
    analysis_result = analysis_output

    movies_to_save = {}
    movies_to_remove = []
    iteration = 0
    max_iterations = 3
    recommendation_history = []

    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")
        
        if iteration == 1:
            rec_message = (
                f"Based on the analysis results: {analysis_result}\n"
                f"Recommend the top 20 movies that best align with the user's preferences. Only provide the movie names in a Python list format."
            )
        else:
            if not movies_to_remove:
                print("All movies have reached a 5-star rating. Process complete.")
                break
            else:
                movies_to_remove_str = json.dumps(movies_to_remove)
                rec_message = (
                    f"Based on the analysis results: {analysis_result}\n"
                    f"Here are the movies to save: {json.dumps(movies_to_save)}\n"
                    f"Remove the following movies from the recommendation list and replace them with new recommendations:\n"
                    f"Movies to remove: {movies_to_remove_str}\n"
                    f"Provide the updated list of 20 movies and specify the new movie(s) added in the format:\n"
                    f"{{'recommended_movies': [...], 'movie_new': [...]}}"
                )

        # 请求 recommendation_agent
        result = entrypoint_agent.initiate_chats([{
            "recipient": rec_agent,
            "message": rec_message,
            "max_turns": 1,
            "summary_method": "last_msg",
            "cache": Cache.disk(cache_seed=seed_num),
        }])
        rec_output = result[-1].chat_history[1]['content'].strip("```json").strip("```").strip()
        print("Recommendation Output:", rec_output)
        try:
            rec_data = json.loads(rec_output)
            recommended_movies = rec_data['recommended_movies']
            movie_new = rec_data['movie_new']  
        except (json.JSONDecodeError, KeyError) as e:
            print("Error parsing recommendation output:", e)
            break

        if iteration == 1:
            movies_to_comment = recommended_movies
        else:
            movies_to_comment = movie_new

        if not movies_to_comment:
            print("No movies to comment on.")
            break


        comment_message = (
            f"Based on the user's analysis results:\n{analysis_result}\n\n"
            f"Suppose you are such a user and here are some movies you've watched:\n{json.dumps(movies_to_comment)}\n\n"
            f"Generate honest and critical comments for each movie as per the system message."
        )
        result = entrypoint_agent.initiate_chats([{
            "recipient": comment_simulator_agent,
            "message": comment_message,
            "max_turns": 1,
            "summary_method": "last_msg",
        }])
        comments_output = result[-1].chat_history[1]['content'].strip("```json").strip("```").strip()

        try:
            print(f"Raw comments output: {comments_output}")  # 调试输出原始数据

            # 检查是否为 JSON 数组
            comments_output_json = re.search(r'\[.*\]', comments_output, re.DOTALL)
            if comments_output_json:
                comments_output_clean = comments_output_json.group(0)
                comments_data = json.loads(comments_output_clean)
            else:
                comments_data = json.loads(comments_output)
                if not isinstance(comments_data, list):
                    comments_data = [comments_data]  
        except Exception as e:
            print("Error parsing comments:", e)
            print(f"Raw comments output: {comments_output}")  
            break

        # 请求 evaluation_agent 对 movie_new 打分
        eval_message = (
            f"Here are the comments for the recommended movies:\n{comments_output_clean}\n\n"
            f"Evaluate these movies from the user's perspective based on the comments provided."
        )
        result = entrypoint_agent.initiate_chats([{
            "recipient": eval_agent,
            "message": eval_message,
            "max_turns": 1,
            "summary_method": "last_msg",
        }])
        eval_output = result[-1].chat_history[1]['content'].strip("```json").strip("```").strip()

        try:
            eval_output_json = json.loads(eval_output)
            evaluations = eval_output_json
        except (json.JSONDecodeError, KeyError) as e:
            print("Error parsing evaluation output:", e)
            break

        judge_message = f"Here are the evaluations:\n{json.dumps(evaluations)}\n\nAs per your instructions, remove movies that are rated less than 4."
        result = entrypoint_agent.initiate_chats([{
            "recipient": judge_agent,
            "message": judge_message,
            "max_turns": 1,
            "summary_method": "last_msg",
        }])
        judge_output = result[-1].chat_history[1]['content'].strip("```json").strip("```").strip()

        try:
            judge_output_json = json.loads(judge_output)
            movies_to_remove = judge_output_json.get('movies_to_remove', [])
            this_round_movies = [eval['movie_title'] for eval in evaluations]

            high_score_movies = [eval for eval in evaluations if eval['evaluation'] >= 4]
            for movie in high_score_movies:
                if movie['movie_title'] not in movies_to_save:
                    movies_to_save[movie['movie_title']] = movie['evaluation']
            movies_to_remove = judge_output_json.get('movies_to_remove', [])

            if judge_output_json.get('process_complete', False):
                print("All movies have been rated 5. Process complete.")
                break

        except Exception as e:
            print("Error parsing judge agent output:", e)
            break

        total_score = sum(movies_to_save.values()) + sum([eval['evaluation'] for eval in evaluations if eval['movie_title'] not in movies_to_save])
        average_score = total_score / 20
        recommendation_history.append({
            'iteration': iteration,
            'recommended_movies': recommended_movies,
            'average_score': average_score
        })
        print("Average Score:", average_score)

    if recommendation_history:
        # 如果有推荐历史，选择平均分最高的推荐结果
        best_recommendation = max(recommendation_history, key=lambda x: x['average_score'])
        print(f"Best recommendation is from iteration {best_recommendation['iteration']} with average score {best_recommendation['average_score']}")
        print("Final Movie List:", best_recommendation['recommended_movies'])
        return best_recommendation['recommended_movies']
    elif movies_to_save:
        # 如果没有推荐历史，但已保存符合条件的电影，直接返回这些电影
        print("Process completed in the first iteration.")
        print("Final Movie List:", list(movies_to_save.keys()))
        return list(movies_to_save.keys())
    else:
        # 如果没有推荐记录也没有保存的电影，说明没有生成任何推荐
        print("No recommendations were generated.")
        return []


# Do not modify this code below.
if __name__ == "__main__":
    main(sys.argv[1])