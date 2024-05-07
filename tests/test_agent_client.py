import os
from typing import Generator, Any
import unittest
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion
from cellium.client import CelliumClient
from pydantic import ConfigDict

load_dotenv()


def fixture() -> CelliumClient:
    client = CelliumClient()
    return client


class TestclientClient(unittest.TestCase):
    openai: OpenAI

    def setUp(self) -> None:
        self.client: CelliumClient = fixture()
        self.openai = OpenAI()
        self.client_api_key: str | None = os.getenv(key="AGENTARTIFICIAL_API_KEY")
        self.client_model = str(object=os.getenv(key="AGENTARTIFICIAL_MODEL"))
        self.client_url = str(object=os.getenv(key="AGENTARTIFICIAL_URL"))
        self.openai_api_key = str(object=os.getenv(key="OPENAI_API_KEY"))
        self.openai_url = str(object=os.getenv(key="OPENAI_URL"))
        self.openai_model = str(object=os.getenv(key="OPENAI_MODEL"))
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ]
        self.model_config = ConfigDict(arbitrary_types_allowed=True)

    def test_check_setup(self) -> None:
        self.assertIsInstance(obj=self.client, cls=CelliumClient)
        self.assertIsInstance(obj=self.openai, cls=OpenAI)
        self.assertEqual(first=self.client.base_url, second=self.client_url)
        self.assertEqual(first=self.client_api_key, second=self.client_api_key)
        self.assertEqual(first=self.client_model, second=self.client_model)
        self.assertEqual(first=self.openai_url, second=self.openai_url)
        self.assertEqual(first=self.openai_api_key, second=self.openai_api_key)
        self.assertEqual(first=self.openai_model, second=str(object=self.openai_model))

    @logger.catch()
    def test_choose_api_key(self) -> None:
        os.environ["OPENAI_API_KEY"] = ""
        result: str = self.client.choose_api_key()
        logger.debug(f"Choose_api_key-1: {result}")
        self.assertEqual(first=result, second=self.client_api_key)

        os.environ["AGENTARTIFICIAL_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = str(object=self.openai_api_key)
        result = self.client.choose_api_key()
        logger.debug(f"Choose_api_key-2: {result}")
        self.assertEqual(first=result, second=self.openai_api_key)

    @logger.catch()
    def test_choose_model(self) -> None:
        os.environ["OPENAI_MODEL"] = ""
        result: str = self.client.choose_model()
        expected_result: str | None = os.getenv(key="AGENTARTIFICIAL_MODEL")
        logger.debug(f"Choose_model-1: {result}")
        self.assertTrue(expr=result)
        self.assertEqual(first=result, second=expected_result)

        os.environ["AGENTARTIFICIAL_MODEL"] = ""
        os.environ["OPENAI_MODEL"] = str(object=self.openai_model)
        result = self.client.choose_model()
        expected_result = "gpt-3.5-turbo"
        logger.debug(f"Choose_model-2 {result}")
        self.assertTrue(expr=result)
        self.assertEqual(first=result, second=expected_result)

    @logger.catch()
    def test_choose_base_url(self) -> None:
        os.environ["OPENAI_URL"] = ""
        result: str = self.client.choose_base_url()
        expected_result: str | None = os.getenv(key="AGENTARTIFICIAL_URL")
        logger.debug(f"Choose_base_url-1 {result}")
        self.assertTrue(expr=result)
        self.assertEqual(first=result, second=expected_result)

        os.environ["AGENTARTIFICIAL_URL"] = ""
        os.environ["OPENAI_URL"] = str(object=self.openai_url)
        result = self.client.choose_base_url()
        expected_result = os.getenv("OPENAI_URL")
        logger.debug(f"Choose_base_url-2 {result}")
        self.assertTrue(expr=result)
        self.assertEqual(first=result, second=expected_result)

    def test_chat_completions(self) -> None:
        self.client.__del__()
        os.environ["AGENTARTIFICIAL_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = str(object=self.openai_api_key)
        self.client.model = str(object=self.client.choose_model())
        response: ChatCompletion = self.openai.chat.completions.create(
            messages=self.messages,  # type: ignore
            model=self.openai_model,
        )
        logger.debug(response)
        self.assertTrue(expr=response)

    def test_generation(self) -> None:
        os.environ["OPENAI_API_KEY"] = ""
        messages: list[dict[str, str]] = self.messages
        response: Generator[Any, Any, None] = self.client.generate(messages=messages)
        logger.debug(response)
        self.assertTrue(expr=response)
