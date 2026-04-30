import os
import json
from dotenv import load_dotenv

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#Tool
def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data['current']

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get current temperature for provided coordinates in celsius.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'latitude': {'type': 'number'},
                    'longitude': {'type': 'number'}
                },
                'required': ['latitude', 'longitude'],
                'additionalProperties': False
            },
            'strict': True
        }
    }
]

#Prompt
system_prompt = 'You are a helpful weather assistant.'

messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': "What's the weather like in Paris today?"}
]

completion = client.chat.completions.create(
    model='gpt-4o',
    messages=messages,
    tools=tools,
)

# print(completion.model_dump())

#Use tools
def call_function(name, args):
    if name == 'get_weather':
        return get_weather(**args)
    
for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {'role': 'tool', 'tool_call_id': tool_call.id, 'content': json.dumps(result)}
    )

#Supply result and call model again
class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )

completion2 = client.beta.chat.completions.parse(
    model='gpt-4o',
    messages=messages,
    tools=tools,
    response_format=WeatherResponse
)

final_response = completion2.choices[0].message.parsed
print(final_response.temperature, final_response.response)