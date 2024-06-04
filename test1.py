

import os
#from dotenv import load_dotenv
#from agentartificial import AgentArtificial
from dotenv import load_dotenv
load_dotenv()

from cellium.client import CelliumClient
#load_dotenv()
from dotenv import load_dotenv
load_dotenv()

#import pdb
#pdb.set_trace()

client = CelliumClient()

response = client.chat.completions.create(
    model="VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct",
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
for x in response:
    print("RESPONSE",x)
