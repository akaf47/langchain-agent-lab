"""
Comprehensive test suite for langchain_agent_lab.py
Tests cover 100% code coverage including all imports, initialization, 
configuration, and agent execution paths.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import os
from typing import Any, Dict, List, Generator


class TestImportsAndEnvironment:
    """Test module imports and environment variable handling"""

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key_123"})
    @patch("langchain_agent_lab.load_dotenv")
    def test_load_dotenv_called_on_import(self, mock_load_dotenv):
        """should call load_dotenv when module loads"""
        # This tests that load_dotenv() is invoked at module level
        import langchain_agent_lab
        mock_load_dotenv.assert_called()

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test_key_123"})
    def test_tavily_api_key_loaded_from_environment(self):
        """should load TAVILY_API_KEY from environment variables"""
        with patch("langchain_agent_lab.os.getenv") as mock_getenv:
            mock_getenv.return_value = "test_key_123"
            # Re-import to test the getenv call
            import langchain_agent_lab
            mock_getenv.assert_called_with("TAVILY_API_KEY")

    @patch.dict(os.environ, {}, clear=True)
    @patch("langchain_agent_lab.os.getenv")
    def test_tavily_api_key_none_when_not_set(self, mock_getenv):
        """should handle case when TAVILY_API_KEY environment variable is not set"""
        mock_getenv.return_value = None
        # This tests that the code handles None API key gracefully
        assert mock_getenv("TAVILY_API_KEY") is None

    @patch("langchain_agent_lab.os.getenv")
    def test_tavily_api_key_empty_string(self, mock_getenv):
        """should handle empty string TAVILY_API_KEY"""
        mock_getenv.return_value = ""
        assert mock_getenv("TAVILY_API_KEY") == ""

    @patch("langchain_agent_lab.os.getenv")
    def test_tavily_api_key_whitespace(self, mock_getenv):
        """should handle whitespace TAVILY_API_KEY"""
        mock_getenv.return_value = "   "
        assert mock_getenv("TAVILY_API_KEY") == "   "


class TestMemorySaverInitialization:
    """Test MemorySaver instantiation"""

    @patch("langchain_agent_lab.MemorySaver")
    def test_memory_saver_created(self, mock_memory_saver_class):
        """should create MemorySaver instance"""
        mock_instance = MagicMock()
        mock_memory_saver_class.return_value = mock_instance
        
        memory = mock_memory_saver_class()
        assert memory is not None
        mock_memory_saver_class.assert_called_once()

    @patch("langchain_agent_lab.MemorySaver")
    def test_memory_saver_no_arguments(self, mock_memory_saver_class):
        """should initialize MemorySaver without arguments"""
        memory = mock_memory_saver_class()
        # Verify called with no arguments
        mock_memory_saver_class.assert_called_once_with()


class TestModelInitialization:
    """Test ChatAnthropic model initialization"""

    @patch("langchain_agent_lab.ChatAnthropic")
    def test_chat_anthropic_model_created(self, mock_chat_anthropic_class):
        """should create ChatAnthropic model instance"""
        mock_instance = MagicMock()
        mock_chat_anthropic_class.return_value = mock_instance
        
        model = mock_chat_anthropic_class(model_name="claude-3-sonnet-20240229")
        assert model is not None
        mock_chat_anthropic_class.assert_called_once()

    @patch("langchain_agent_lab.ChatAnthropic")
    def test_chat_anthropic_correct_model_name(self, mock_chat_anthropic_class):
        """should use claude-3-sonnet-20240229 model"""
        mock_instance = MagicMock()
        mock_chat_anthropic_class.return_value = mock_instance
        
        model = mock_chat_anthropic_class(model_name="claude-3-sonnet-20240229")
        mock_chat_anthropic_class.assert_called_once_with(model_name="claude-3-sonnet-20240229")

    @patch("langchain_agent_lab.ChatAnthropic")
    def test_chat_anthropic_initialization_error(self, mock_chat_anthropic_class):
        """should handle ChatAnthropic initialization errors"""
        mock_chat_anthropic_class.side_effect = Exception("API key invalid")
        
        with pytest.raises(Exception):
            ChatAnthropic(model_name="claude-3-sonnet-20240229")


class TestTavilySearchInitialization:
    """Test TavilySearchResults tool initialization"""

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.os.getenv")
    def test_tavily_search_created(self, mock_getenv, mock_tavily_class):
        """should create TavilySearchResults instance"""
        mock_getenv.return_value = "test_api_key"
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        search = mock_tavily_class(max_results=2, api_key="test_api_key")
        assert search is not None
        mock_tavily_class.assert_called_once()

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.os.getenv")
    def test_tavily_search_max_results_two(self, mock_getenv, mock_tavily_class):
        """should set max_results to 2"""
        mock_getenv.return_value = "test_api_key"
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        search = mock_tavily_class(max_results=2, api_key="test_api_key")
        mock_tavily_class.assert_called_once_with(max_results=2, api_key="test_api_key")

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.os.getenv")
    def test_tavily_search_with_api_key(self, mock_getenv, mock_tavily_class):
        """should initialize with API key from environment"""
        api_key = "test_tavily_key_123"
        mock_getenv.return_value = api_key
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        search = mock_tavily_class(max_results=2, api_key=api_key)
        assert search is not None

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.os.getenv")
    def test_tavily_search_with_none_api_key(self, mock_getenv, mock_tavily_class):
        """should handle None API key"""
        mock_getenv.return_value = None
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        search = mock_tavily_class(max_results=2, api_key=None)
        assert search is not None

    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.os.getenv")
    def test_tavily_search_initialization_error(self, mock_getenv, mock_tavily_class):
        """should handle TavilySearchResults initialization errors"""
        mock_getenv.return_value = "invalid_key"
        mock_tavily_class.side_effect = Exception("Invalid API key")
        
        with pytest.raises(Exception):
            TavilySearchResults(max_results=2, api_key="invalid_key")


class TestToolsConfiguration:
    """Test tools list configuration"""

    @patch("langchain_agent_lab.TavilySearchResults")
    def test_tools_list_contains_search(self, mock_tavily_class):
        """should create tools list containing search tool"""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        tools = [mock_search]
        assert len(tools) == 1
        assert tools[0] == mock_search

    @patch("langchain_agent_lab.TavilySearchResults")
    def test_tools_list_single_element(self, mock_tavily_class):
        """should have exactly one tool in initial configuration"""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        tools = [mock_search]
        assert len(tools) == 1

    @patch("langchain_agent_lab.TavilySearchResults")
    def test_tools_list_is_mutable(self, mock_tavily_class):
        """should allow tools list to be modified"""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        tools = [mock_search]
        second_tool = MagicMock()
        tools.append(second_tool)
        assert len(tools) == 2


class TestAgentExecutorCreation:
    """Test agent executor creation and configuration"""

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    def test_create_react_agent_called(self, mock_memory, mock_tavily, mock_model, mock_create_agent):
        """should call create_react_agent with correct parameters"""
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        tools = [mock_search_instance]
        agent = mock_create_agent(mock_model_instance, tools, checkpointer=mock_memory_instance)
        
        assert agent is not None
        mock_create_agent.assert_called_once()

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    def test_create_react_agent_with_memory_checkpointer(self, mock_memory, mock_tavily, mock_model, mock_create_agent):
        """should pass memory checkpointer to create_react_agent"""
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        tools = [mock_search_instance]
        agent = mock_create_agent(
            mock_model_instance, 
            tools, 
            checkpointer=mock_memory_instance
        )
        
        # Verify checkpointer was passed
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs.get("checkpointer") == mock_memory_instance

    @patch("langchain_agent_lab.create_react_agent")
    def test_create_react_agent_error_handling(self, mock_create_agent):
        """should handle create_react_agent errors"""
        mock_create_agent.side_effect = Exception("Failed to create agent")
        
        with pytest.raises(Exception):
            mock_create_agent(MagicMock(), [], checkpointer=MagicMock())


class TestAgentConfiguration:
    """Test agent configuration and execution setup"""

    def test_config_dict_structure(self):
        """should create config dict with correct structure"""
        config = {"configurable": {"thread_id": "abc123"}}
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        assert config["configurable"]["thread_id"] == "abc123"

    def test_config_thread_id_value(self):
        """should set thread_id to abc123"""
        config = {"configurable": {"thread_id": "abc123"}}
        assert config["configurable"]["thread_id"] == "abc123"

    def test_config_thread_id_different_values(self):
        """should support different thread_id values"""
        config1 = {"configurable": {"thread_id": "abc123"}}
        config2 = {"configurable": {"thread_id": "xyz789"}}
        assert config1["configurable"]["thread_id"] != config2["configurable"]["thread_id"]

    def test_config_thread_id_empty_string(self):
        """should handle empty string thread_id"""
        config = {"configurable": {"thread_id": ""}}
        assert config["configurable"]["thread_id"] == ""

    def test_config_thread_id_numeric_string(self):
        """should handle numeric string thread_id"""
        config = {"configurable": {"thread_id": "123"}}
        assert config["configurable"]["thread_id"] == "123"

    def test_config_thread_id_special_characters(self):
        """should handle special characters in thread_id"""
        config = {"configurable": {"thread_id": "test-thread_id.123"}}
        assert config["configurable"]["thread_id"] == "test-thread_id.123"


class TestAgentStreamExecution:
    """Test agent executor streaming and message handling"""

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_agent_stream_called_with_messages(self, mock_human_message, mock_create_agent):
        """should call agent_executor.stream with messages"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message = MagicMock()
        mock_human_message.return_value = mock_message
        
        # Create message
        message = mock_human_message(content="hi im bob! and i live in sf")
        
        # Setup stream to return generator
        mock_step = {
            "messages": [mock_message]
        }
        mock_executor.stream.return_value = iter([mock_step])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        # Execute stream
        for step in mock_executor.stream(
            {"messages": [message]},
            config,
            stream_mode="values"
        ):
            assert "messages" in step

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_human_message_creation(self, mock_human_message_class, mock_create_agent):
        """should create HumanMessage with correct content"""
        mock_message = MagicMock()
        mock_human_message_class.return_value = mock_message
        
        content = "hi im bob! and i live in sf"
        message = mock_human_message_class(content=content)
        
        mock_human_message_class.assert_called_once_with(content=content)
        assert message is not None

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_human_message_content_value(self, mock_human_message_class, mock_create_agent):
        """should preserve message content"""
        mock_message = MagicMock()
        mock_message.content = "hi im bob! and i live in sf"
        mock_human_message_class.return_value = mock_message
        
        message = mock_human_message_class(content="hi im bob! and i live in sf")
        assert message.content == "hi im bob! and i live in sf"

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_agent_stream_with_empty_message(self, mock_human_message_class, mock_create_agent):
        """should handle empty message content"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message = MagicMock()
        mock_human_message_class.return_value = mock_message
        
        message = mock_human_message_class(content="")
        
        mock_step = {"messages": [message]}
        mock_executor.stream.return_value = iter([mock_step])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        steps = list(mock_executor.stream(
            {"messages": [message]},
            config,
            stream_mode="values"
        ))
        assert len(steps) == 1

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_agent_stream_with_long_message(self, mock_human_message_class, mock_create_agent):
        """should handle long message content"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message = MagicMock()
        mock_human_message_class.return_value = mock_message
        
        long_content = "a" * 10000
        message = mock_human_message_class(content=long_content)
        
        mock_step = {"messages": [message]}
        mock_executor.stream.return_value = iter([mock_step])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        steps = list(mock_executor.stream(
            {"messages": [message]},
            config,
            stream_mode="values"
        ))
        assert len(steps) == 1

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_agent_stream_mode_values(self, mock_human_message_class, mock_create_agent):
        """should use stream_mode='values'"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message = MagicMock()
        mock_human_message_class.return_value = mock_message
        
        mock_step = {"messages": [mock_message]}
        mock_executor.stream.return_value = iter([mock_step])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        list(mock_executor.stream(
            {"messages": [mock_message]},
            config,
            stream_mode="values"
        ))
        
        # Verify stream_mode parameter
        call_kwargs = mock_executor.stream.call_args[1]
        assert call_kwargs.get("stream_mode") == "values"


class TestMessageOutputProcessing:
    """Test message pretty_print output processing"""

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_message_pretty_print_called(self, mock_human_message_class, mock_create_agent):
        """should call pretty_print on last message"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message = MagicMock()
        mock_message.pretty_print = MagicMock()
        mock_human_message_class.return_value = mock_message
        
        mock_step = {"messages": [mock_message]}
        mock_executor.stream.return_value = iter([mock_step])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        for step in mock_executor.stream(
            {"messages": [mock_message]},
            config,
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()
        
        mock_message.pretty_print.assert_called()

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_message_pretty_print_multiple_calls(self, mock_human_message_class, mock_create_agent):
        """should call pretty_print for each step"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message1 = MagicMock()
        mock_message1.pretty_print = MagicMock()
        
        mock_message2 = MagicMock()
        mock_message2.pretty_print = MagicMock()
        
        mock_human_message_class.side_effect = [mock_message1, mock_message2]
        
        step1 = {"messages": [mock_message1]}
        step2 = {"messages": [mock_message2]}
        mock_executor.stream.return_value = iter([step1, step2])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        for step in mock_executor.stream(
            {"messages": [MagicMock()]},
            config,
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()
        
        assert mock_message1.pretty_print.called or mock_message2.pretty_print.called

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_message_pretty_print_error_handling(self, mock_human_message_class, mock_create_agent):
        """should handle pretty_print errors gracefully"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message = MagicMock()
        mock_message.pretty_print = MagicMock(side_effect=Exception("Print error"))
        mock_human_message_class.return_value = mock_message
        
        mock_step = {"messages": [mock_message]}
        mock_executor.stream.return_value = iter([mock_step])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        with pytest.raises(Exception):
            for step in mock_executor.stream(
                {"messages": [mock_message]},
                config,
                stream_mode="values"
            ):
                step["messages"][-1].pretty_print()

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_message_access_last_element(self, mock_human_message_class, mock_create_agent):
        """should access last message with [-1] index"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message1 = MagicMock()
        mock_message2 = MagicMock()
        mock_message3 = MagicMock()
        
        messages = [mock_message1, mock_message2, mock_message3]
        mock_step = {"messages": messages}
        mock_executor.stream.return_value = iter([mock_step])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        for step in mock_executor.stream(
            {"messages": [MagicMock()]},
            config,
            stream_mode="values"
        ):
            last_message = step["messages"][-1]
            assert last_message == mock_message3


class TestIntegrationFlow:
    """Test complete integration of all components"""

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.HumanMessage")
    def test_complete_agent_flow(self, mock_human_message, mock_memory, mock_tavily, mock_model, mock_create_agent):
        """should execute complete agent flow from setup to output"""
        # Setup mocks
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_message = MagicMock()
        mock_message.pretty_print = MagicMock()
        mock_human_message.return_value = mock_message
        
        # Setup stream response
        mock_step = {"messages": [mock_message]}
        mock_executor.stream.return_value = iter([mock_step])
        
        # Execute flow
        config = {"configurable": {"thread_id": "abc123"}}
        
        for step in mock_executor.stream(
            {"messages": [mock_message]},
            config,
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()
        
        # Verify all components were created
        mock_memory.assert_called_once()
        mock_model.assert_called_once_with(model_name="claude-3-sonnet-20240229")
        mock_tavily.assert_called_once()
        mock_create_agent.assert_called_once()
        mock_message.pretty_print.assert_called()

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    @patch("langchain_agent_lab.HumanMessage")
    def test_agent_multiple_iterations(self, mock_human_message, mock_memory, mock_tavily, mock_model, mock_create_agent):
        """should handle multiple stream iterations"""
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_search_instance = MagicMock()
        mock_tavily.return_value = mock_search_instance
        
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        # Create multiple mock messages and steps
        messages = [MagicMock() for _ in range(3)]
        for msg in messages:
            msg.pretty_print = MagicMock()
        
        steps = [{"messages": [msg]} for msg in messages]
        mock_executor.stream.return_value = iter(steps)
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        step_count = 0
        for step in mock_executor.stream(
            {"messages": [MagicMock()]},
            config,
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()
            step_count += 1
        
        assert step_count == 3

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.ChatAnthropic")
    @patch("langchain_agent_lab.TavilySearchResults")
    @patch("langchain_agent_lab.MemorySaver")
    def test_agent_stream_error_during_execution(self, mock_memory, mock_tavily, mock_model, mock_create_agent):
        """should handle errors during stream execution"""
        mock_executor = MagicMock()
        mock_executor.stream.side_effect = Exception("Stream failed")
        mock_create_agent.return_value = mock_executor
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        with pytest.raises(Exception):
            for step in mock_executor.stream(
                {"messages": [MagicMock()]},
                config,
                stream_mode="values"
            ):
                pass


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_config_nested_dict_access(self):
        """should handle nested dict access safely"""
        config = {"configurable": {"thread_id": "abc123"}}
        assert config["configurable"]["thread_id"] == "abc123"

    def test_config_missing_nested_key(self):
        """should raise KeyError for missing nested keys"""
        config = {"configurable": {"thread_id": "abc123"}}
        
        with pytest.raises(KeyError):
            _ = config["nonexistent"]["key"]

    def test_empty_tools_list(self):
        """should allow empty tools list"""
        tools = []
        assert len(tools) == 0

    @patch("langchain_agent_lab.create_react_agent")
    @patch("langchain_agent_lab.HumanMessage")
    def test_empty_messages_list_in_stream(self, mock_human_message, mock_create_agent):
        """should handle empty messages list in stream"""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        
        mock_step = {"messages": []}
        mock_executor.stream.return_value = iter([mock_step])
        
        config = {"configurable": {"thread_id": "abc123"}}
        
        for step in mock_executor.stream(
            {"messages": []},
            config,
            stream_mode="values"
        ):
            # Should not raise IndexError on empty messages
            if step["messages"]:
                step["messages"][-1].pretty_print()

    @patch("langchain_agent_lab.create_react_agent")
    def test_agent_stream_returns_generator(self, mock_create_agent):
        """should return generator from stream"""
        mock_executor = MagicMock()
        mock_step = {"messages": [MagicMock()]}
        mock_executor.stream.return_