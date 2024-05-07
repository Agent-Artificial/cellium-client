import os
from dotenv import load_dotenv
from agent.artificial import AgentArtificial


load_dotenv()


def get_agent():
    """
    Instantiates the AgentArtificial client and sets its API key and model based on the values in the .env file.
    If the .env file does not specify an API key or model, the client falls back to using the default values.

    Returns:
        AgentArtificial: The instantiated AgentArtificial client.
    """
    client = AgentArtificial()
    # The client checks your .env file for which API key and model to use
    # Default is AgentArtificial with a fall back of OpenAI.
    client.api_key = client.choose_api_key()

    # Use the choose method to auto select.
    client.model = str(os.getenv("AGENTARTIFICIAL_MODEL"))
    # Or you can load them normally
    client.url = str(os.getenv("AGENTARTIFICIAL_URL"))

    # now the client works the same as openai's client.
    return client


def example():
    """
    Calls the get_agent function to instantiate an AgentArtificial client.
    Creates a chat completion using the instantiated agent client.
    The completion is based on a set of messages and returns the content of the first choice from the completion response.
    """
    agent = get_agent()
    # Creating a chat completion is the same as openai
    response = agent.chat.completions.create(
        model=agent.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ],
    )
    return response["data"]["choices"][0]["message"]["content"]


if __name__ == "__main__":
    get_agent()
