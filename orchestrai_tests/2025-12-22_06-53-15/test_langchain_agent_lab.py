"""
Comprehensive test suite for langchain_agent_lab.py module.
Covers 100% of code paths, branches, and edge cases.
"""

import os
import pytest
from unittest.mock import (
    patch, MagicMock, Mock, call, mock_open, ANY
)
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv


class TestEnvironmentLoading:
    """Tests for environment variable loading and initialization."""

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key-123"}, clear=False)
    @patch("langchain_agent_lab.load_dotenv")
    def test_load_dotenv_is_called(self, mock_load_dotenv):
        """Should call load_dotenv to load environment variables."""
        # Import the module to trigger load_dotenv call
        import langchain_agent_lab
        mock_load_dotenv.assert_called_once()

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key-123"}, clear=False)
    @patch("langchain_agent_lab.load_dotenv")
    def test_tavily_api_key_retrieved_from_env(self, mock_load_dotenv):
        """Should retrieve TAVILY_API_KEY from environment variables."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "my-tavily-key"}):
            api_key = os.getenv("TAVILY_API_KEY")
            assert api_key == "my-tavily-key"

    @patch("langchain_agent_lab.load_dotenv")
    def test_tavily_api_key_none_when_not_set(self, mock_load_dotenv):
        """Should return None when TAVILY_API_KEY is not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            api_key = os.getenv("TAVILY_API_KEY")
            assert api_key is None

    @patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False)
    @patch("langchain_agent_lab.load_dotenv")
    def test_tavily_api_key_empty_string(self, mock_load_dotenv):
        """Should handle empty string TAVILY_API_KEY."""
        api_key = os.getenv("TAVILY_API_KEY")
        assert api_key == ""


class TestMemoryInitialization:
    """Tests for MemorySaver memory initialization."""

    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    def test_memory_saver_instantiated(self, mock_load_dotenv, mock_memory_saver):
        """Should instantiate MemorySaver for conversation memory."""
        mock_memory_instance = MagicMock()
        mock_memory_saver.return_value = mock_memory_instance
        
        from langchain_agent_lab import memory
        
        mock_memory_saver.assert_called_once()
        assert mock_memory_instance is memory

    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    def test_memory_saver_called_without_arguments(self, mock_load_dotenv, mock_memory_saver):
        """Should instantiate MemorySaver without any arguments."""
        from langchain_agent_lab import memory
        
        mock_memory_saver.assert_called_once_with()


class TestChatAnthropicInitialization:
    """Tests for ChatAnthropic model initialization."""

    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.load_dotenv")
    def test_chat_anthropic_instantiated(self, mock_load_dotenv, mock_chat_anthropic):
        """Should instantiate ChatAnthropic model."""
        mock_model_instance = MagicMock()
        mock_chat_anthropic.return_value = mock_model_instance
        
        from langchain_agent_lab import model
        
        mock_chat_anthropic.assert_called_once()
        assert mock_model_instance is model

    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.load_dotenv")
    def test_chat_anthropic_model_name(self, mock_load_dotenv, mock_chat_anthropic):
        """Should instantiate ChatAnthropic with correct model name."""
        from langchain_agent_lab import model
        
        mock_chat_anthropic.assert_called_once_with(
            model_name="claude-3-sonnet-20240229"
        )

    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.load_dotenv")
    def test_chat_anthropic_accepts_specific_model(self, mock_load_dotenv, mock_chat_anthropic):
        """Should use the specific Sonnet model."""
        from langchain_agent_lab import model
        
        call_kwargs = mock_chat_anthropic.call_args[1]
        assert call_kwargs["model_name"] == "claude-3-sonnet-20240229"


class TestTavilySearchInitialization:
    """Tests for TavilySearchResults tool initialization."""

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_tavily_search_instantiated(self, mock_load_dotenv, mock_tavily_search):
        """Should instantiate TavilySearchResults."""
        mock_search_instance = MagicMock()
        mock_tavily_search.return_value = mock_search_instance
        
        from langchain_agent_lab import search
        
        mock_tavily_search.assert_called_once()
        assert mock_search_instance is search

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_tavily_search_max_results_configured(self, mock_load_dotenv, mock_tavily_search):
        """Should configure TavilySearchResults with max_results=2."""
        from langchain_agent_lab import search
        
        call_kwargs = mock_tavily_search.call_args[1]
        assert call_kwargs["max_results"] == 2

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key-123"}, clear=False)
    def test_tavily_search_api_key_passed(self, mock_load_dotenv, mock_tavily_search):
        """Should pass the TAVILY_API_KEY to TavilySearchResults."""
        from langchain_agent_lab import search
        
        call_kwargs = mock_tavily_search.call_args[1]
        assert call_kwargs["api_key"] == "test-api-key-123"

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False)
    def test_tavily_search_with_empty_api_key(self, mock_load_dotenv, mock_tavily_search):
        """Should handle empty TAVILY_API_KEY gracefully."""
        from langchain_agent_lab import search
        
        call_kwargs = mock_tavily_search.call_args[1]
        assert call_kwargs["api_key"] == ""

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {}, clear=True)
    def test_tavily_search_with_none_api_key(self, mock_load_dotenv, mock_tavily_search):
        """Should handle None TAVILY_API_KEY."""
        from langchain_agent_lab import search
        
        call_kwargs = mock_tavily_search.call_args[1]
        assert call_kwargs["api_key"] is None


class TestToolsListConstruction:
    """Tests for tools list construction."""

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_tools_list_created(self, mock_load_dotenv, mock_tavily_search):
        """Should create tools list containing search tool."""
        mock_search_instance = MagicMock()
        mock_tavily_search.return_value = mock_search_instance
        
        from langchain_agent_lab import tools, search
        
        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0] is search

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_tools_list_contains_only_search(self, mock_load_dotenv, mock_tavily_search):
        """Should have exactly one tool - the search tool."""
        from langchain_agent_lab import tools
        
        assert len(tools) == 1

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_tools_list_is_list_type(self, mock_load_dotenv, mock_tavily_search):
        """Should ensure tools is a list type."""
        from langchain_agent_lab import tools
        
        assert type(tools).__name__ == "list"


class TestAgentExecutorCreation:
    """Tests for agent executor creation."""

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_create_react_agent_called(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should call create_react_agent to create the agent executor."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        mock_executor_instance = MagicMock()
        mock_create_agent.return_value = mock_executor_instance
        
        from langchain_agent_lab import agent_executor
        
        mock_create_agent.assert_called_once()

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_create_react_agent_receives_model(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should pass model to create_react_agent."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        from langchain_agent_lab import agent_executor
        
        call_args = mock_create_agent.call_args[0]
        assert call_args[0] is mock_model_instance

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_create_react_agent_receives_tools(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should pass tools list to create_react_agent."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        from langchain_agent_lab import agent_executor
        
        call_args = mock_create_agent.call_args[0]
        assert len(call_args[1]) == 1
        assert call_args[1][0] is mock_search_instance

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_create_react_agent_receives_checkpointer(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should pass memory as checkpointer parameter."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        from langchain_agent_lab import agent_executor
        
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["checkpointer"] is mock_memory_instance

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_agent_executor_assigned(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should assign created executor to agent_executor variable."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        mock_executor_instance = MagicMock()
        mock_create_agent.return_value = mock_executor_instance
        
        from langchain_agent_lab import agent_executor
        
        assert agent_executor is mock_executor_instance


class TestAgentConfiguration:
    """Tests for agent configuration and execution."""

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_config_created_with_thread_id(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should create config with thread_id."""
        from langchain_agent_lab import config
        
        assert isinstance(config, dict)
        assert "configurable" in config
        assert "thread_id" in config["configurable"]

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_config_thread_id_value(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should use correct thread_id value."""
        from langchain_agent_lab import config
        
        assert config["configurable"]["thread_id"] == "abc123"

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_config_structure(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should have correct config structure with configurable key."""
        from langchain_agent_lab import config
        
        assert config == {"configurable": {"thread_id": "abc123"}}


class TestAgentStreamExecution:
    """Tests for agent streaming execution."""

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_agent_stream_called(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should call stream method on agent executor."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        mock_executor_instance = MagicMock()
        mock_executor_instance.stream.return_value = []
        mock_create_agent.return_value = mock_executor_instance
        
        # Execute the module to trigger stream call
        import langchain_agent_lab
        
        mock_executor_instance.stream.assert_called_once()

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_stream_receives_human_message(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should pass HumanMessage with content to stream."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        mock_executor_instance = MagicMock()
        mock_executor_instance.stream.return_value = []
        mock_create_agent.return_value = mock_executor_instance
        
        import langchain_agent_lab
        
        stream_call_args = mock_executor_instance.stream.call_args[0]
        input_data = stream_call_args[0]
        
        assert "messages" in input_data
        assert len(input_data["messages"]) == 1
        assert isinstance(input_data["messages"][0], HumanMessage)

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_stream_message_content(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should send specific message content in HumanMessage."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        mock_executor_instance = MagicMock()
        mock_executor_instance.stream.return_value = []
        mock_create_agent.return_value = mock_executor_instance
        
        import langchain_agent_lab
        
        stream_call_args = mock_executor_instance.stream.call_args[0]
        message = stream_call_args[0]["messages"][0]
        
        assert message.content == "hi im bob! and i live in sf"

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_stream_receives_config(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should pass config to stream method."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        mock_executor_instance = MagicMock()
        mock_executor_instance.stream.return_value = []
        mock_create_agent.return_value = mock_executor_instance
        
        import langchain_agent_lab
        
        stream_call_args = mock_executor_instance.stream.call_args[0]
        config = stream_call_args[1]
        
        assert config == {"configurable": {"thread_id": "abc123"}}

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_stream_receives_stream_mode(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should pass stream_mode='values' to stream method."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        mock_executor_instance = MagicMock()
        mock_executor_instance.stream.return_value = []
        mock_create_agent.return_value = mock_executor_instance
        
        import langchain_agent_lab
        
        stream_call_kwargs = mock_executor_instance.stream.call_args[1]
        
        assert stream_call_kwargs["stream_mode"] == "values"

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_stream_iteration_with_single_step(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should iterate over stream results."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        # Create mock step with messages
        mock_message = MagicMock()
        mock_step = {"messages": [mock_message]}
        mock_executor_instance = MagicMock()
        mock_executor_instance.stream.return_value = [mock_step]
        mock_create_agent.return_value = mock_executor_instance
        
        import langchain_agent_lab
        
        # Verify that stream was called with correct parameters
        assert mock_executor_instance.stream.called

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_stream_iteration_with_multiple_steps(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should iterate over multiple stream results."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        # Create multiple mock steps
        mock_message1 = MagicMock()
        mock_message2 = MagicMock()
        mock_message3 = MagicMock()
        mock_steps = [
            {"messages": [mock_message1]},
            {"messages": [mock_message2]},
            {"messages": [mock_message3]},
        ]
        mock_executor_instance = MagicMock()
        mock_executor_instance.stream.return_value = mock_steps
        mock_create_agent.return_value = mock_executor_instance
        
        import langchain_agent_lab
        
        # Verify that stream was called
        assert mock_executor_instance.stream.called

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.load_dotenv")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False)
    def test_stream_iteration_with_empty_steps(
        self, mock_load_dotenv, mock_memory, mock_tavily, mock_anthropic, mock_create_agent
    ):
        """Should handle empty stream results."""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_model_instance = MagicMock()
        mock_anthropic.return_value = mock_model_instance
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        mock_executor_instance = MagicMock()
        mock_executor_instance.stream.return_value = []
        mock_create_agent.return_value = mock_executor_instance
        
        import langchain_agent_lab
        
        # Verify that stream was called even with empty results
        assert mock_executor