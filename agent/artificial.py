import os
import json
from httpx import HTTPError
import typing
from abc import ABC, abstractmethod
from openai import OpenAI
from websocket import WebSocket
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional, Union, Any, Callable

from dotenv import load_dotenv

AbstractSetIntStr = typing.Union[typing.AbstractSet[int], typing.AbstractSet[str]]
MappingIntStrAny = typing.Union[typing.Mapping[int, Any], typing.Mapping[str, Any]]

load_dotenv()

AGENTARTIFICIAL_URL = str(os.getenv("AGENTARTIFICIAL_URL"))
OPENAI_URL = str(os.getenv("OPENAI_URL"))

AGENTARTIFICIAL_API_KEY = str(os.getenv("AGENTARTIFICIAL_API_KEY"))
OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))

AGENTARTIFICIAL_MODEL = str(os.getenv("AGENTARTIFICIAL_MODEL"))
OPENAI_MODEL = str(os.getenv("OPENAI_MODEL"))

openai = OpenAI(api_key=AGENTARTIFICIAL_API_KEY, base_url=AGENTARTIFICIAL_URL)


class Completion(BaseModel):
    create: Callable

    def __init_subclass__(cls, create, **kwargs: Any):
        cls.create = create(**kwargs)


class Chat(BaseModel):
    completions: Completion

    def __init_subclass__(cls, completions, **kwargs: Any):
        cls.completions = completions(**kwargs)


class Agent(BaseModel, ABC):
    openai: OpenAI
    agent_url: str
    agent_api_key: str
    agent_model: str
    openai_url: str
    openai_api_key: str
    openai_model: str
    web_socket: Optional[WebSocket]
    openai: OpenAI
    is_inference: bool
    base_url: str
    api_key: str
    model: str
    completion: Completion
    chat: Chat
    is_inference: bool
    model_config = ConfigDict(arbitrary_types_allowed=True)
    __pydantic_fields_set__ = {"is_inference"}

    @abstractmethod
    def choose_model(self) -> Union[str, None]:
        """
        Choose model from OPENAI_MODEL or AGENTARTIFICIAL_MODEL
        """

    @abstractmethod
    def choose_base_url(self) -> Union[str, None]:
        """
        Choose base_url from OPENAI_URL or AGENTARTIFICIAL_URL
        """

    @abstractmethod
    def choose_api_key(self) -> Union[str, None]:
        """
        Choose api_key from OPENAI_API_KEY or AGENTARTIFICIAL_API_KEY
        """


class AgentArtificial(Agent):
    """
    The AgentArtificial class is a child class of the OpenAI class that enables
    the user to make inferences with the Agent Artificial API endpoints for
    inference. The class will automatically configure the client to hit Agent
    Artificial endpoints.

    The class has several key features:

    1. It can use the Agent Artificial API endpoints for inference by setting
    the environment variable AGENTARTIFICIAL_URL and AGENTARTIFICIAL_API_KEY.
    2. It can use the OpenAI API endpoints for inference by setting the
    environment variable OPENAI_URL and OPENAI_API_KEY.
    3. It can choose the model to use for inference by setting the environment
    variable AGENTARTIFICIAL_MODEL or OPENAI_MODEL.
    4. It can create a chat session by using the create method with a list of
    messages.
    5. It can close the websocket connection when the object is deleted by
    using the __del__ method.

    Example:
        import os
        from dotenv import load_dotenv
        from agentartificial import AgentArtificial

        load_dotenv()

        client = AgentArtificial()


        response = client.chat.completions.create(
            messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'Who won the world series in 2020?'},
                    {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
                    {'role': 'user', 'content': 'Where was it played?'},
                ]
        )
        message = response['data']['choices'][0]['message']['content']
    """

    def __init__(self) -> None:
        """
        Initializes the AgentArtificial object by setting various attributes such as agent_url, agent_api_key, agent_model, openai_url, openai_api_key, openai_model, web_socket, openai, is_inference, url, api, and model.
        """
        self.agent_url = AGENTARTIFICIAL_URL
        self.agent_api_key = AGENTARTIFICIAL_API_KEY
        self.agent_model = AGENTARTIFICIAL_MODEL
        self.openai_url = OPENAI_URL
        self.openai_api_key = OPENAI_API_KEY
        self.openai_model = OPENAI_MODEL
        self.is_inference = True if self.agent_url else False
        self.base_url = self.choose_base_url()
        self.api_key = self.choose_api_key()
        self.model = self.choose_model()
        self.openai = self.get_openai_client() or openai
        self.web_socket = None
        self.completion = Completion(create=self.create)
        self.chat = Chat(completions=self.completion)
        super().__init__(
            agent_url=self.agent_url,
            agent_api_key=self.agent_api_key,
            agent_model=self.agent_model,
            openai_url=self.openai_url,
            openai_api_key=self.openai_api_key,
            openai_model=self.openai_model,
            is_inference=self.is_inference,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            openai=self.openai,
            web_socket=self.web_socket,
            completion=self.completion,
            chat=self.chat,
        )

    def create(self, **kwargs):  # -> Generator[Any, Any, None] | Any:
        if self.is_inference:
            messages: List[Dict[str, str]] = kwargs["messages"]
            return self.generate(messages)
        else:
            if self.openai:
                return self.openai.chat.completions.create(**kwargs)
        raise ValueError("No inference available")

    def generate(self, messages: List[Dict[str, str]]):
        prompt = self.create_prompt(messages)
        url = (f"{self.agent_url}/api/v2/generate",)
        preflight = {
            "url": url,
            "type": "openinference_session",
            "max_size": 4096,
        }
        payload = {
            "type": "openinference_session",
            "prompt": prompt,
        }
        try:
            if not self.web_socket:
                raise HTTPError("Websocket connection not established")
            self.web_socket.send(json.dumps(preflight))
            self.web_socket.send(json.dumps(payload))
            while self.web_socket.receive():
                message = self.web_socket.receive()
                if message is None:
                    break
                yield json.loads(message)
            self.web_socket.close()
        except HTTPError as e:
            raise HTTPError("HTTP request failed") from e

    def create_prompt(self, messages: List[Dict[str, str]]):
        """
        Creates a prompt string from a list of dictionaries representing messages.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries representing messages. Each dictionary should have the keys 'role' and 'content', where 'role' is the role of the message sender and 'content' is the content of the message.

        Returns:
            str: The prompt string created from the messages.

        Example:
            create_prompt([
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Who won the world series in 2020?'},
                {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
                {'role': 'user', 'content': 'Where was it played?'}
            ])
            # Returns:
            # 'system\nYou are a helpful assistant.\nuser\nWho won the world series in 2020?\nassistant\nThe Los Angeles Dodgers won the World Series in 2020.\nuser\nWhere was it played?'
        """
        prompt = ""
        for message in messages:
            prompt += f"{message['role']}\n"
            prompt += f"{message['content']}\n"

        return prompt

    def connect(self) -> WebSocket:
        """
        Connects to a WebSocket using the provided environment variables and creates a WebSocket object.
        Returns the WebSocket object.
        """
        self.web_socket = WebSocket(environ=os.environ, socket=WebSocket, rfile=None)
        return self.web_socket

    def get_openai_client(self) -> OpenAI:
        self.openai = OpenAI()
        self.openai.api_key = self.choose_api_key()
        if not self.openai:
            raise ValueError("Client did not load correctly")
        return self.openai

    def choose_api_key(self) -> str:
        if self.is_inference:
            return str(os.getenv("AGENTARTIFICIAL_API_KEY"))
        return str(os.getenv("OPENAI_API_KEY"))

    def choose_model(self) -> str:
        if self.is_inference:
            return str(os.getenv("AGENTARTIFICIAL_MODEL"))
        return str(os.getenv("OPENAI_MODEL"))

    def choose_base_url(self) -> str:
        if self.is_inference:
            return str(os.getenv("AGENTARTIFICIAL_URL"))
        return str(os.getenv("OPENAI_URL"))

    def __del__(self) -> None:
        if self.web_socket:
            self.web_socket.close()

        if self.openai:
            self.openai.close()
