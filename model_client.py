import os
import openai
import anthropic
import google.generativeai as g
from google.ai import generativelanguage as glm
import requests

def anthropic_send_messages(messages, model="claude-3-opus-20240229", max_tokens=4096):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": os.environ['ANTHROPIC_API_KEY'],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception if the request was unsuccessful

    return response.json()

class Client:
    def query_model(self, messages, user_message):
        raise NotImplementedError


class OpenAIClient(Client):
    def __init__(self, model):
        self.client = openai.OpenAI(api_key=os.environ[f"OPENAI_API_KEY"], base_url=os.environ[f"OPENAI_BASE_URL"])
        self.model = model

    def query_model(self, messages, user_message):
        messages.append({"role": "user", "content": user_message})
        print(f"USER message:\n{messages[-1]['content']}")
        response_message = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        ).choices[0].message
        response_message = dict(response_message)
        response_message = {"role": "assistant", "content": response_message['content']}
        messages.append(response_message)
        print(f"RESPONSE message:\n{messages[-1]['content']}")
        return messages, response_message['content']

class GoogleClient:
    def __init__(self, model):
        g.configure(api_key=os.environ[f"GOOGLE_API_KEY"])
        self.client = g.GenerativeModel(model)
        self.model = model

    def query_model(self, messages, user_message):
        part = g.types.content_types.to_part(user_message)
        content = glm.Content(role="user", parts=[part])
        messages.append(content)
        print(f"USER message:\n{messages[-1].parts[0].text}")
        response_message = self.client.generate_content(messages)
        messages.append(response_message.candidates[0].content)
        print(f"RESPONSE message:\n{messages[-1].parts[0].text}")
        return messages, response_message.parts[0].text

class AnthropicClient:
    def __init__(self, model):
        self.model = model

    def query_model(self, messages, user_message):
        messages.append({"role": "user", "content": user_message})
        print(f"USER message:\n{messages[-1]['content']}")
        response_message = anthropic_send_messages(messages, model=self.model)
        response_message = {"role": "assistant", "content": response_message['content'][0]['text']}
        messages.append(response_message)
        print(f"RESPONSE message:\n{messages[-1]['content']}")
        return messages, response_message['content']

