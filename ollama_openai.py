from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionUserMessageParam, ChatCompletionChunk
from typing import Iterable
import json
from pydantic import BaseModel, Field
import instructor

class RelevanceScore(BaseModel):
    relevance:int = Field(..., description="The relevance score of the input")



if __name__ == '__main__':
    modelname = "openhermes"
    base_url = "http://localhost:11434/v1/"
    client = OpenAI(base_url=base_url, api_key="api")
    msg = ChatCompletionUserMessageParam(content="Hello, how are you today?", role="user")
    messages = [msg]  
    chat_response: ChatCompletion = client.chat.completions.create(model=modelname, messages=messages, stream=False)
    print(chat_response.choices[0].message.content)
    # Next w e want to try streaming the response
    chat_streaming_response: Iterable[ChatCompletionChunk] = client.chat.completions.create(model=modelname, messages=messages, stream=True)
    for fragment in chat_streaming_response:
        print(fragment.choices[0].delta.content)
    # Now let's try to get a json response
    json_payload = {
        "input": "The tiger is blue",
        "topic": "colorful animals",
        "task": "rate relevance 0 for no and 1 for yes",
        "response_schema": "{ \"relevance\": \"int\"}",
    }
    msg = ChatCompletionUserMessageParam(content=json.dumps(json_payload), role="user")
    messages = [msg]
    chat_response = client.chat.completions.create(model=modelname, messages=messages, stream=False, response_format={"type": "json_object"})
    content = chat_response.choices[0].message.content
    # This might fail
    relevance = RelevanceScore(**json.loads(content))
    print(relevance.relevance)
    # Now we want to enfore the response schema by using instructor
    patched_client = instructor.patch(OpenAI(base_url=base_url, api_key="api"), mode=instructor.Mode.JSON)
    chat_response: RelevanceScore = patched_client.chat.completions.create(model=modelname, messages=messages, response_format={"type": "json_object"}, response_model=RelevanceScore,  max_retries=3)
    print(chat_response.relevance) # Might still fail actaully with floats and ints being an issue