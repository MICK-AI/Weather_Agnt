#CHAIN OF THOUGHT PROMPTING
import requests
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)



def get_weather(city):
    url = f"https://wttr.in/{city}?format=%25c+%25t"

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return response.text

available_tools = {
    "get_weather": get_weather
}



SYSTEM_PROMPT=""" You are expert ai assistant in resolving user queries using
chain of thought.
you work on START,PLAN, and OUTPUT steps.
you need to first PLAN what needs to be done ,The Plan can be multiple steps.
once you think enough PLAN has been done, finally you can give an OUTPUT
Return exactly ONE valid JSON object per response.
Never return multiple JSON objects.
Do not include markdown or text outside the JSON.
Rules:
-Strictly follow the given JSON output format
-only run one step at a time
- The sequence of steps is START(where user gives an input),PLAN(that
can be multiple times) and finally OUTPUT (which is going to the displayed
to the user)
- for every tool call w ait for the observe step which is output from the called tool.

Output JSON Format:
{"step":"start" | "PLAN" } "OUTPUT"| "TOOL" ,"content":"string" , " tool":"string", "input":"string"}

Available Tools:
- get_weather(City: str): Take city name as input string and returns the weather info about the cityy.

Example 1:
START: Hey, can you solve 2+3 *5 /12
PLAN : {"step":"PLAN":"content": "Seems like user is interested in math problem"}
PLAN : {"step":"PLAN":"content  ": "Looking at the problem, we should solve this using BODMAS method"}
PLAN : {"step":"PLAN":"content": "Yes, the BODMAS is the correct thing to be used here"}
PLAN : {"step":"PLAN":"content": "First we multiply 3*5 which is 15"}
PLAN : {"step":"PLAN":"content": "Now the equation is 2+15/10"}
PLAN : {"step":"PLAN":"content": "we must perform divide that is 15/10=1.5"}
PLAN : {"step":"PLAN":"content": "Now the equation is 2+1.5"}
PLAN : {"step":"PLAN":"content": "Now, finally lets perform the add , 3.5"}
PLAN : {"step":"PLAN":"content": "Great, Problem solved, answer is 3.5"}
OUTPUT:{"step":"OUTPUT":"content":"3.5}

Example 2:
START: What is the weather of Delhi
PLAN : {"step":"PLAN":"content": "Seems like user is interested in weather of Delhi in India"}
PLAN : {"step":"PLAN":"content  ": "Lets see if we have any available tools from the list of available tools"}
PLAN : {"step":"PLAN":"content": "Great! we have the get_weather tool available for this query"}
PLAN : {"step":"PLAN":"content": "I need to call get_weather tool for delhi as input city"}
PLAN : {"step":"TOOL": "Tool": "get_weather","Input": "Delhi"}
PLAN : {"step":"TOOL": "OBSERVE": "get_weather","Output": "the temperature of delhi is cludy with 20 degree celcius"}
PLAN : {"step":"PLAN":"content": "Great we got weather info about"}
OUTPUT:{"step":"OUTPUT":"content":"The current weather in delhi is 20 degree celcius with claudy weather"}

IMPORTANT:
Return ONLY ONE JSON object per API response.
Never return multiple JSON objects.
Generate exactly one step at a time.
If the input is unclear, return OUTPUT asking the user to clarify.

Rules for execution:
- Generate only ONE step per response.
- PLAN steps must contain new information, not repeat previous steps.
- Use a maximum of 3 PLAN steps.
- After 1-3 PLAN steps, you MUST return OUTPUT.
- For simple greetings or unclear input, return OUTPUT immediately.
- Never repeat a previous PLAN step.
"""


print("\n\n\n")

message_history = [
    {"role":"system","content":SYSTEM_PROMPT}
]
while True:
    user_query = input("What do you wanna know?")
    message_history.append({"role":"user","content":user_query})


    plan_count = 0

    while True:
        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            response_format={"type": "json_object"},
            messages=message_history
        )

        raw_result = response.choices[0].message.content

        try:
            parsed_result = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            print("Invalid JSON received from model:")
            print(raw_result)
            break

        # Save the model's response to conversation history
        message_history.append({
            "role": "assistant",
            "content": raw_result
        })

        step = parsed_result.get("step", "").upper()
        content = parsed_result.get("content", "")

        if step == "START":
            print("Starting LLM Loop :-)", content)
            continue

        if step == "TOOL":
                tool_to_call = parsed_result.get("tool")
                tool_input = parsed_result.get("input")
                tool_response = available_tools[tool_to_call](tool_input)
                print(f"-_-' {tool_to_call} ({tool_input}) = {tool_response}")
        
                message_history.append({"role": "user","content":json.dumps(
                    {"step":"Observe","tool":tool_to_call, "input" :tool_input, "output": tool_response }
                )})
                continue

        elif step == "PLAN":
            plan_count += 1
            print("Thinking:", content)

            if plan_count >= 3:
                message_history.append({
                    "role": "user",
                    "content": "Stop planning and return the final OUTPUT now."
                })
            else:
                message_history.append({
                    "role": "user",
                    "content": "Continue to the next step. Return one JSON object only."
                })


        elif step == "OUTPUT":
            print("Solution:", content)
            break

        else:
            print("Unexpected response:", parsed_result)
            break



    #     message_history.append
    #    # parsed_result = json.loads(raw_result)
    #     #if isinstance(parsed_result, list):
    #      #   parsed_result = parsed_result[0]

    #     if parsed_result.get("step")=="START":
    #         print("Starting LLM Loop :-) ",parsed_result.get("content"))
    #         continue
        
    #     if parsed_result.get("step")=="PLAN":
    #         print("Thinking",parsed_result.get("content"))
    #         continue

    #     if parsed_result.get("step")=="OUTPUT":
    #         print("Solution",parsed_result.get("content"))
    #         break

    print("\n\n\n")