# Script used for evaluation of generated response and ground truth by prompting ChatGPT with the result
import json
from openai import OpenAI
import os
import re
import pandas as pd

data_dict = {
             "Math": "Wup/Results/Math4_8_16_results.json",
             "Linguistic": "Wup/Results/Linguistic16_results.json",
             "CrossDomain": "Cross/mathR1.json"
            }



linguistic_res = json.loads(open(data_dict["Linguistic"]).read())
#math_res = json.loads(open(data_dict["Math"]).read())


# Generate the results needed for prompt formatting
def result_formatting(res):
    dict = {}

    # Iterate over R
    for key in res:
        list = []
        for entry in res[key]:
            
            list.append({"generated_response": res[key][entry]["response"], "ground_truth": res[key][entry]["ground_truth"]})

        dict[key] = list
    return dict

## Generation of prompts sent to ChatGPT API for benchmark results
# Generate prompts for linguistic calibration
def generate_ling_prompt(prompt):
    gen_resp = prompt["generated_response"]
    ground_truth = prompt["ground_truth"]

    return f'''I will provide to you a generated reponse and a ground truth aka desired reponse. 
                Provide a score of the linguistic level of the generated response compared to the ground truth 
                based on the following metrics:\n
                -  Lexical Diversity\n
                - Syntactic Complexity\n
                -  Clarity and Cohesion\n
                -  Use of Figurative Language or Stylistic Elements\n
                -  Readability\n
                -  Engagement\n
                Assume the level of the desired response corresponds to 100. Give all metrics equal weight. Provide only the score between 0 and 100 as your response.
                \n
                Here are the generated response and the ground truth:
                \n
                *Generated response*
                {gen_resp}
                \n
                *Ground truth*
                {ground_truth}
                '''


# Generate prompts for math
def generate_math_prompt(prompt):
    gen_resp = prompt["generated_response"]
    gen_resp =gen_resp.split("[/INST]")[1]
    ground_truth = prompt["ground_truth"]
    return f'''
        I will provide to you a generated reponse and a ground truth aka desired reponse. 
        Provide a score of the mathematical level of the generated response compared to the ground truth 
        based on the following metrics:\n
        - Clarity and Setup of the Problem\n
        - Complexity of Arithmetic\n
        - Conceptual Understanding\n
        - Precision and Attention to Detail\n
        - Logical Structure\n
        - Accessibility\n
        - Challenge Level\n
        \n
        Assume the level of the desired response corresponds to 100. Award 40 out of 100 points if the correct 
        answer was reached. Award the remaining 60 points among the metrics, giving equal weight to all. Provide 
        only the score between 0 and 100 as your response.
        \n
        Here are the generated response and the ground truth:
        \n
        *Generated response*
        {gen_resp}
        \n
        *Ground truth*
        {ground_truth}
        '''

# Version of R
#math_prompts = result_formatting(math_res)
ling_prompts = result_formatting(linguistic_res)

#math_dict = {}
#for R in math_prompts:
#   math_dict[R] = [generate_math_prompt(math_prompt) for math_prompt in math_prompts[R]]


ling_dict = {}
for R in ling_prompts:
    ling_dict[R] = [generate_ling_prompt(ling_prompt) for ling_prompt in ling_prompts[R]]

# Print the first prompt for each R
print(ling_dict["FineTuned_Linguistic_R16"][0])


# Prints first prompt for each R
#print(math_dict["FineTuned_Math_R1"][0])
#print(ling_dict["FineTuned_Linguistic_R1"][0])




api_key = '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'

def generate_scores(prompts):
    client = OpenAI(
        api_key = api_key
        )
    scores = []
    for prompt in prompts:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":prompt}],
            stream=True,
            temperature=0 # Keeping consistency
        )
        stream_chunks = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                stream_chunks += chunk.choices[0].delta.content
        try:
            res = re.findall(r'\d+', stream_chunks)
            res = list(map(int, res))
            scores.append(res[0])  
        except:
            scores.append(0)

    return scores    
            
""""
for r in math_dict:
    math_scores = generate_scores(math_dict[r])

    # Created dataframe to store the results of math
    df = pd.DataFrame({"Math": math_scores})

    # Save the results to a csv file
    df.to_csv(f"Evaluation/Results1/Math/{r}_scores.csv", index=False)

"""
for r in ling_dict:
    ling_scores = generate_scores(ling_dict[r])

    # Created dataframe to store the results of math and linguistic prompts
    df = pd.DataFrame({"Linguistic": ling_scores})

    # Save the results to a csv file
    df.to_csv(f"Evaluation/Results1/Linguistic/{r}_scores.csv", index=False)
