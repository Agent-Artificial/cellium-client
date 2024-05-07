# Agent Artificial OpenAI Client

This library enables OpenAI standard client to interface with the Agent Artificial API endpoints for inference. 

## Instillation 

`pip install agentartificial`

## Usage

Import `AgentArtificial` and instantiate a new instance and pass the `OpenAI` class into the agent client. It will automatically configure the client to hit Agent Artificial endpoints. 

Example
```
import os
from dotenv import load_dotenv
from agentartificial import AgentArtificial

load_dotenv()

client = AgentArtificial()



response = client.chat.completions.create(
    model=client.agent_model,
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ]
)
message = response['data']['choices'][0]['message']['content']


```
