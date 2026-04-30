import os
import json
from dotenv import load_dotenv

from openai import OpenAI
from pydantic import BaseModel, Field
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#Knowledge base retrieval tool
def search_kb(question: str):
    """for testing purposes, we don't search. We return the whole knowledge base"""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'knowledge_base.json'
    with open(data_path, 'r') as f:
        return json.load(f)

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'search_kb',
            'description': "Get the answer to the user's question from the knowledge base.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'question': {'type': 'string'}
                },
                'required': ['question'],
                'additionalProperties': False
            },
            'strict': True
        }
    }
]

#Call model
system_prompt = 'You are a helpful assistant that answers questions from the knowledge base.'

messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': 'What is the return policy?'}
]

completion = client.chat.completions.create(
    model='gpt-4o',
    messages=messages,
    tools=tools
)

# print(completion.model_dump())

#Call tool
def call_function(name, args):
    if name == 'search_kb':
        return search_kb(**args)

for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {'role': 'tool', 'tool_call_id': tool_call.id, 'content': json.dumps(result)}
    )

#Supply result and call model again
class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")

completion2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    response_format=KBResponse,
)

final_response = completion2.choices[0].message.parsed
print(final_response.answer, final_response.source)

#Ex question that doesn't trigger the tool
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

completion_3 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

print(completion_3.choices[0].message.content)