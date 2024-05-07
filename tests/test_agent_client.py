import os
import unittest
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from agent.artificial import AgentArtificial
from pydantic import ConfigDict

load_dotenv()


def fixture() -> AgentArtificial:
    agent = AgentArtificial()
    return agent


class TestAgentClient(unittest.TestCase):
    openai: OpenAI

    def setUp(self):
        self.agent = fixture()
        self.openai = OpenAI()
        self.agent_api_key = os.getenv("AGENTARTIFICIAL_API_KEY")
        self.agent_model = str(os.getenv("AGENTARTIFICIAL_MODEL"))
        self.agent_url = str(os.getenv("AGENTARTIFICIAL_URL"))
        self.openai_api_key = str(os.getenv("OPENAI_API_KEY"))
        self.openai_url = str(os.getenv("OPENAI_URL"))
        self.openai_model = str(os.getenv("OPENAI_MODEL"))
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ]
        self.model_config = ConfigDict(arbitrary_types_allowed=True)

    def test_check_setup(self):
        self.assertIsInstance(self.agent, AgentArtificial)
        self.assertIsInstance(self.openai, OpenAI)
        self.assertEqual(self.agent.base_url, self.agent_url)
        self.assertEqual(self.agent_api_key, self.agent_api_key)
        self.assertEqual(self.agent_model, self.agent_model)
        self.assertEqual(self.openai_url, self.openai_url)
        self.assertEqual(self.openai_api_key, self.openai_api_key)
        self.assertEqual(self.openai_model, str(self.openai_model))

    @logger.catch()
    def test_choose_api_key(self):
        os.environ["OPENAI_API_KEY"] = ""
        result = self.agent.choose_api_key()
        logger.debug(f"Choose_api_key-1: {result}")
        self.assertEqual(result, self.agent_api_key)

        os.environ["AGENTARTIFICIAL_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = str(self.openai_api_key)
        result = self.agent.choose_api_key()
        logger.debug(f"Choose_api_key-2: {result}")
        self.assertEqual(result, self.openai_api_key)

    @logger.catch()
    def test_choose_model(self):
        os.environ["OPENAI_MODEL"] = ""
        result = self.agent.choose_model()
        expected_result = os.getenv("AGENTARTIFICIAL_MODEL")
        logger.debug(f"Choose_model-1: {result}")
        self.assertTrue(result)
        self.assertEqual(result, expected_result)

        os.environ["AGENTARTIFICIAL_MODEL"] = ""
        os.environ["OPENAI_MODEL"] = str(self.openai_model)
        result = self.agent.choose_model()
        expected_result = "gpt-3.5-turbo"
        logger.debug(f"Choose_model-2 {result}")
        self.assertTrue(result)
        self.assertEqual(result, expected_result)

    @logger.catch()
    def test_choose_base_url(self):
        os.environ["OPENAI_URL"] = ""
        result = self.agent.choose_base_url()
        expected_result = os.getenv("AGENTARTIFICIAL_URL")
        logger.debug(f"Choose_base_url-1 {result}")
        self.assertTrue(result)
        self.assertEqual(result, expected_result)

        os.environ["AGENTARTIFICIAL_URL"] = ""
        os.environ["OPENAI_URL"] = str(self.openai_url)
        result = self.agent.choose_base_url()
        expected_result = os.getenv("OPENAI_URL")
        logger.debug(f"Choose_base_url-2 {result}")
        self.assertTrue(result)
        self.assertEqual(result, expected_result)

    def test_chat_completions(self):
        self.agent.__del__()
        os.environ["AGENTARTIFICIAL_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = str(self.openai_api_key)
        self.agent.model = str(self.agent.choose_model())
        response = self.openai.chat.completions.create(
            messages=self.messages,  # type: ignore
            model=self.openai_model,
        )
        logger.debug(response)
        self.assertTrue(response)

    def test_generation(self):
        os.environ["OPENAI_API_KEY"] = ""
        messages = self.messages
        response = self.agent.generate(messages=messages)
        logger.debug(response)
        self.assertTrue(response)
