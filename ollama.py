from ollama import chat, Message
from typing import TypedDict, Optional
import json
from pydantic import BaseModel

class RelevanceScore(BaseModel):
    relevance: int


class ChatResponse(TypedDict):
    message: Message
    model: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int
    created_at: str #yyyy-mm-ddThh:mm:ss.sssZ

class StreamingResponse(TypedDict):
    message: Message
    model: str
    created_at: str #yyyy-mm-ddThh:mm:ss.sssZ
    done: bool
    total_duration: Optional[int] # Set if done
    load_duration: Optional[int] # Set if done
    prompt_eval_duration: Optional[int] # Set if done
    eval_count: Optional[int] # Set if done
    eval_duration: Optional[int] # Set if done




if __name__ == "__main__":
    msg = Message(role="user", content="Hello, how are you today?")
    messages = [msg]
    model = "openhermes"
    response = chat(model=model, messages=messages, stream=False)
    response = ChatResponse(**response) 
    print(response["message"]["content"]) 
    # Next we want to try streaming the response
    response = chat(model=model, messages=messages, stream=True)
    for fragment in response:
        print(fragment)
        fragment = StreamingResponse(**fragment)
        print(fragment["message"]["content"])

    # Next we want to try ollama json 
    json_payload = {
        "input": "The tiger is blue",
        "topic": "colorful animals",
        "task": "rate relevance 0 for no and 1 for yes",
        "response_schema": "{ \"relevance\": \"int\"}",
    }
    msg = Message(role="user", content=json.dumps(json_payload))
    messages = [msg]
    response = chat(model=model, messages=messages, stream=False, format="json")
    content = response["message"]["content"]
    # This might fail
    relevance = RelevanceScore(**json.loads(content))
    print(relevance.relevance)