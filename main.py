from typing import Dict, List
import numpy as np 
from autogen import ConversableAgent
import sys
import os
import re
import json
import pandas as pd
from autogen import Cache
import pickle
import ast

from config.prompts import prompts

seed_num = 123

with open('config/meta_data.json', 'r') as f:
    meta_data = json.load(f)
    
with open('draft_retrieval/combined_retrieval.pkl', 'rb') as f:
    retrieval = pickle.load(f)
    
with open('config/history_reviews.json', 'r') as f:
    history = json.load(f)
    
def get_retrieval(user_id: str) -> Dict[str, Dict]:
    return {item:
        meta_data[item]['title'] for item in retrieval[user_id]
    }
        
def fetch_reviews(user_id, item_id):
    return history[user_id][item_id]

def fetch_data(user_id: str, item_sequence: list) -> Dict[str, Dict]:
    return {
        item: {
            "item meta information": meta_data[item],
            "user review": fetch_reviews(user_id, item)
        } for item in item_sequence
    }
    

def main(user_query: str):

    entrypoint_agent_system_message = ""
    fetchdata_agent_system_message = prompts["fetchdata_agent_system_message"]
    retrieval_agent_system_message = prompts["retrieval_agent_system_message"]
    analysis_agent_system_message = prompts["analysis_agent_system_message"]
    rec_agent_system_message = prompts["rec_agent_system_message"]
    comment_simulator_system_message = prompts["comment_simulator_system_message"]
    eval_agent_system_message = prompts["eval_agent_system_message"]
    judge_agent_system_message = prompts["judge_agent_system_message"]
    
    
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    entrypoint_agent = ConversableAgent("entrypoint_agent",
                                        system_message=entrypoint_agent_system_message,
                                        llm_config=llm_config,
                                        human_input_mode='NEVER')
    entrypoint_agent.register_for_execution(name="fetch_data")(fetch_data)
    entrypoint_agent.register_for_execution(name='get_retrieval')(get_retrieval)

    
    fetchdata_agent = ConversableAgent("fetchdata_agent",
                                       system_message=fetchdata_agent_system_message,
                                       llm_config=llm_config,
                                       max_consecutive_auto_reply=1,
                                       human_input_mode='NEVER')
    fetchdata_agent.register_for_llm(name="fetch_data", description="Fetches the item information.")(fetch_data)
    
    retrieval_agent = ConversableAgent("retrieval_agent",
                                       system_message=retrieval_agent_system_message,
                                       llm_config=llm_config,
                                       max_consecutive_auto_reply=1,
                                       human_input_mode='NEVER')
    retrieval_agent.register_for_llm(name="get_retrieval", description="Get draft model retrieval.")(get_retrieval)

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
    user_history_info = datafetch_chat_result.chat_history[2]['content']
    
    retrieval_chat_result = entrypoint_agent.initiate_chat(retrieval_agent, message=user_query, max_turns=2)
    retrieval_list = retrieval_chat_result.chat_history[2]['content']

    result = entrypoint_agent.initiate_chats([
        {
            "recipient": analysis_agent,
            "message": user_history_info,
            "max_turns": 1,
            "summary_method": "last_msg",
        }
    ])
    analysis_output = result[-1].chat_history[1]['content']
    analysis_result = analysis_output

    items_to_save = {}
    items_to_remove = []
    iteration = 0
    max_iterations = 2

    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")
        
        if iteration == 1:
            rec_message = (
                f"Based on the analysis results: {analysis_result}\n"
                f"And the cadidate set: {retrieval_list}\n"
                "Recommend the top 20 products that best align with the user's preferences from the analysis result, must consider all the product."
            )
        else:
            if not items_to_remove:
                print("All products have good. Process complete.")
                return recommended_items
                break
            else:
                items_to_remove_str = json.dumps(items_to_remove)
                rec_message = (
                    f"Based on the analysis results: {analysis_result}\n"
                    f"Here are the products to save: {json.dumps(items_to_save)}\n"
                    f"Remove the following products from the recommendation list and replace them with new recommendations that will have positive evaluation scores:\n"
                    f"Movies to remove: {items_to_remove_str}\n"
                    f"Provide the updated list of 20 products in recommended_items (must have length 20!!!) and specify the new product(s) in the item_new:\n"
                    f"{{'recommended_items': [...], 'item_new': [...]}}"
                )

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
            recommended_items = [list(rec.keys())[0] for rec in rec_data['recommended_items']]
            
            item_new = rec_data['item_new']  

        except (json.JSONDecodeError, KeyError) as e:
            print("Error parsing recommendation output:", e)
            return []
            # break
    

        if iteration == 1:
             items_to_comment = recommended_items
        else:
             items_to_comment = item_new

        if not items_to_comment:
             print("No items to comment on.")
             break

        
        comment_message = (
            f"Based on the user's review history:\n{user_history_info}\n\n"
            f"You are tasked with writing honest, critical, and realistic comments for each of the following products, reflecting the user's preferences, tone, and style:\n\n"
            f"{json.dumps(items_to_comment)}\n\n"
            f"**Instructions for Writing Comments:**\n"
            f"1. Write comments as if you are the user, ensuring the tone aligns with their review history.\n"
            f"2. Provide balanced feedback by highlighting both strengths and weaknesses of each product.\n"
            f"3. Avoid overly generic statements; include specific aspects like quality, usability, price, or features where applicable.\n"
            f"4. Do not use the word 'but' to maintain a smoother and more natural tone.\n"
            f"5. Ensure each comment is concise, insightful, and relevant to the product's description and the user's preferences.\n\n"
  
            f"Generate comments for ALL products listed above, ensuring they reflect the user's voice accurately."
        )

        result = entrypoint_agent.initiate_chats([{
             "recipient": comment_simulator_agent,
             "message": comment_message,
             "max_turns": 1,
             "summary_method": "last_msg",
         }])
        comments_output = result[-1].chat_history[1]['content'].strip("```json").strip("```").strip()
        
        try:
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

        judge_message = f"Here are the evaluations:\n{evaluations}\n\n Remove items that have extreme negative comments and rating(-2)."
        result = entrypoint_agent.initiate_chats([{
             "recipient": judge_agent,
             "message": judge_message,
             "max_turns": 1,
             "summary_method": "last_msg",
         }])
        judge_output = result[-1].chat_history[1]['content'].strip("```json").strip("```").strip()

        try:
             judge_output_json = json.loads(judge_output)
             items_to_remove = judge_output_json.get('items_to_remove', [])

             if judge_output_json.get('process_complete', False):
                 print("No item to remove. Process complete.")
                 break

        except Exception as e:
             print("Error parsing judge agent output:", e)
             break
        
    '''
    if items_to_save:
         # 如果没有推荐历史，但已保存符合条件的电影，直接返回这些电影
         print("Process completed in the first iteration.")
         print("Final items List:", list(items_to_save.keys()))
         return list(items_to_save.keys())
    else:
         # 如果没有推荐记录也没有保存的电影，说明没有生成任何推荐
         print("No recommendations were generated.")
         return []
    '''
    return recommended_items

# Do not modify this code below.
if __name__ == "__main__":
    res = main("Please recommend for user 'AHV6QCNBJNSGLATP56JAWJ3C4G2A', who had purchased [B07NPWK167 B07SW7D6ZR B07WNBZQGT B082NKQ4ZT B083TLNBJJ B087D7MVHB B088FBNQXW B085NYYLQ8 B08BZ1RHPS B0B2L218H2 B08HXQ3T9K B08KWN77LW]")
    print(res)