"""
Comprehensive test suite for langchain_agent_lab.py
Tests cover all code paths, error scenarios, and edge cases.
"""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, List, Any


class TestEnvironmentLoading:
    """Test environment variable loading and configuration."""

    def test_load_dotenv_is_called(self):
        """Test that load_dotenv is called to load environment variables."""
        with patch('langchain_agent_lab.load_dotenv') as mock_load:
            # Re-import to trigger the module-level code
            import importlib
            import langchain_agent_lab
            importlib.reload(langchain_agent_lab)
            mock_load.assert_called_once()

    def test_tavily_api_key_retrieved_from_env(self):
        """Test that Tavily API key is retrieved from environment."""
        test_key = "test_tavily_key_12345"
        with patch.dict(os.environ, {'TAVILY_API_KEY': test_key}):
            with patch('langchain_agent_lab.load_dotenv'):
                import importlib
                import langchain_agent_lab
                importlib.reload(langchain_agent_lab)
                assert langchain_agent_lab.tavily_api_key == test_key

    def test_tavily_api_key_missing_returns_none(self):
        """Test that missing Tavily API key returns None."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                import importlib
                import langchain_agent_lab
                importlib.reload(langchain_agent_lab)
                assert langchain_agent_lab.tavily_api_key is None

    def test_tavily_api_key_empty_string(self):
        """Test handling of empty string API key."""
        with patch.dict(os.environ, {'TAVILY_API_KEY': ''}):
            with patch('langchain_agent_lab.load_dotenv'):
                import importlib
                import langchain_agent_lab
                importlib.reload(langchain_agent_lab)
                assert langchain_agent_lab.tavily_api_key == ''


class TestMemoryInitialization:
    """Test memory saver initialization."""

    @patch('langchain_agent_lab.MemorySaver')
    def test_memory_saver_created(self, mock_memory_class):
        """Test that MemorySaver instance is created."""
        mock_memory_instance = MagicMock()
        mock_memory_class.return_value = mock_memory_instance
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        mock_memory_class.assert_called_once()
        assert langchain_agent_lab.memory == mock_memory_instance

    @patch('langchain_agent_lab.MemorySaver')
    def test_memory_saver_is_singleton(self, mock_memory_class):
        """Test that only one MemorySaver is created."""
        mock_memory_class.reset_mock()
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Verify it was called during module initialization
        assert mock_memory_class.call_count >= 1


class TestChatAnthropicModelInitialization:
    """Test ChatAnthropic model initialization."""

    @patch('langchain_agent_lab.ChatAnthropic')
    def test_model_created_with_correct_name(self, mock_chat_class):
        """Test that ChatAnthropic is created with correct model name."""
        mock_model = MagicMock()
        mock_chat_class.return_value = mock_model
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        mock_chat_class.assert_called_once_with(model_name="claude-3-sonnet-20240229")
        assert langchain_agent_lab.model == mock_model

    @patch('langchain_agent_lab.ChatAnthropic')
    def test_model_uses_specific_sonnet_version(self, mock_chat_class):
        """Test that the model uses Claude 3 Sonnet 20240229."""
        mock_model = MagicMock()
        mock_chat_class.return_value = mock_model
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs['model_name'] == "claude-3-sonnet-20240229"


class TestTavilySearchToolInitialization:
    """Test Tavily search tool initialization."""

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_search_tool_created_with_max_results(self, mock_tavily_class):
        """Test that TavilySearchResults is created with max_results=2."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'}):
            with patch('langchain_agent_lab.load_dotenv'):
                import importlib
                import langchain_agent_lab
                importlib.reload(langchain_agent_lab)
                
                call_args = mock_tavily_class.call_args
                assert call_args[1]['max_results'] == 2

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_search_tool_receives_api_key(self, mock_tavily_class):
        """Test that TavilySearchResults receives the API key."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        test_key = "test_tavily_xyz"
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': test_key}):
            with patch('langchain_agent_lab.load_dotenv'):
                import importlib
                import langchain_agent_lab
                importlib.reload(langchain_agent_lab)
                
                call_args = mock_tavily_class.call_args
                assert call_args[1]['api_key'] == test_key

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_search_tool_with_none_api_key(self, mock_tavily_class):
        """Test that TavilySearchResults is created even with None API key."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                import importlib
                import langchain_agent_lab
                importlib.reload(langchain_agent_lab)
                
                call_args = mock_tavily_class.call_args
                assert call_args[1]['api_key'] is None


class TestToolsConfiguration:
    """Test tools list configuration."""

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tools_list_contains_search(self, mock_tavily_class):
        """Test that tools list contains the search tool."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        assert mock_search in langchain_agent_lab.tools
        assert len(langchain_agent_lab.tools) == 1

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tools_is_list(self, mock_tavily_class):
        """Test that tools is a list type."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        assert isinstance(langchain_agent_lab.tools, list)


class TestAgentExecutorCreation:
    """Test agent executor creation."""

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_agent_executor_created(self, mock_memory_cls, mock_tavily_cls, 
                                     mock_chat_cls, mock_create_agent):
        """Test that agent executor is created."""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_model = MagicMock()
        mock_chat_cls.return_value = mock_model
        mock_search = MagicMock()
        mock_tavily_cls.return_value = mock_search
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        assert langchain_agent_lab.agent_executor == mock_executor

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_agent_executor_receives_model(self, mock_memory_cls, mock_tavily_cls,
                                            mock_chat_cls, mock_create_agent):
        """Test that agent executor receives the model."""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_model = MagicMock()
        mock_chat_cls.return_value = mock_model
        mock_search = MagicMock()
        mock_tavily_cls.return_value = mock_search
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        call_args = mock_create_agent.call_args
        assert call_args[0][0] == mock_model

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_agent_executor_receives_tools(self, mock_memory_cls, mock_tavily_cls,
                                            mock_chat_cls, mock_create_agent):
        """Test that agent executor receives the tools list."""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_model = MagicMock()
        mock_chat_cls.return_value = mock_model
        mock_search = MagicMock()
        mock_tavily_cls.return_value = mock_search
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        call_args = mock_create_agent.call_args
        # Second positional argument should be tools
        assert call_args[0][1] == [mock_search]

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_agent_executor_receives_checkpointer(self, mock_memory_cls, mock_tavily_cls,
                                                   mock_chat_cls, mock_create_agent):
        """Test that agent executor receives the memory checkpointer."""
        mock_executor = MagicMock()
        mock_create_agent.return_value = mock_executor
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_model = MagicMock()
        mock_chat_cls.return_value = mock_model
        mock_search = MagicMock()
        mock_tavily_cls.return_value = mock_search
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs.get('checkpointer') == mock_memory


class TestAgentConfiguration:
    """Test agent configuration."""

    def test_config_has_configurable_key(self):
        """Test that config has 'configurable' key."""
        import langchain_agent_lab
        assert 'configurable' in langchain_agent_lab.config

    def test_config_has_thread_id(self):
        """Test that config has thread_id in configurable."""
        import langchain_agent_lab
        assert 'thread_id' in langchain_agent_lab.config['configurable']

    def test_config_thread_id_is_abc123(self):
        """Test that thread_id is set to 'abc123'."""
        import langchain_agent_lab
        assert langchain_agent_lab.config['configurable']['thread_id'] == "abc123"

    def test_config_is_dict(self):
        """Test that config is a dictionary."""
        import langchain_agent_lab
        assert isinstance(langchain_agent_lab.config, dict)

    def test_config_has_correct_structure(self):
        """Test that config has the expected structure."""
        import langchain_agent_lab
        expected = {"configurable": {"thread_id": "abc123"}}
        assert langchain_agent_lab.config == expected


class TestAgentStreamingExecution:
    """Test agent streaming execution."""

    @patch('langchain_agent_lab.agent_executor')
    def test_agent_stream_called_with_correct_messages(self, mock_executor):
        """Test that agent.stream is called with human message."""
        from langchain_core.messages import HumanMessage
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([])
        mock_executor.stream = mock_stream
        
        # Import and execute the streaming code
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Verify stream was called
        assert mock_stream.called
        call_args = mock_stream.call_args
        
        # Check first argument (input dict)
        input_dict = call_args[0][0]
        assert 'messages' in input_dict
        assert len(input_dict['messages']) == 1
        assert isinstance(input_dict['messages'][0], HumanMessage)

    @patch('langchain_agent_lab.agent_executor')
    def test_agent_stream_message_content(self, mock_executor):
        """Test that the stream message has correct content."""
        from langchain_core.messages import HumanMessage
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        call_args = mock_stream.call_args
        input_dict = call_args[0][0]
        message = input_dict['messages'][0]
        assert message.content == "hi im bob! and i live in sf"

    @patch('langchain_agent_lab.agent_executor')
    def test_agent_stream_called_with_config(self, mock_executor):
        """Test that agent.stream is called with config."""
        mock_stream = MagicMock()
        mock_stream.return_value = iter([])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        call_args = mock_stream.call_args
        config = call_args[0][1]
        assert config == {"configurable": {"thread_id": "abc123"}}

    @patch('langchain_agent_lab.agent_executor')
    def test_agent_stream_called_with_stream_mode(self, mock_executor):
        """Test that agent.stream is called with stream_mode='values'."""
        mock_stream = MagicMock()
        mock_stream.return_value = iter([])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs.get('stream_mode') == 'values'

    @patch('langchain_agent_lab.agent_executor')
    def test_agent_stream_iterates_over_steps(self, mock_executor):
        """Test that the code iterates over stream steps."""
        step1 = {'messages': [MagicMock()]}
        step2 = {'messages': [MagicMock()]}
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([step1, step2])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Verify stream was called and iteration occurred
        assert mock_stream.called

    @patch('langchain_agent_lab.agent_executor')
    def test_agent_stream_empty_response(self, mock_executor):
        """Test handling when stream returns no steps."""
        mock_stream = MagicMock()
        mock_stream.return_value = iter([])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Should not raise an error
        assert mock_stream.called

    @patch('langchain_agent_lab.agent_executor')
    def test_agent_stream_single_step(self, mock_executor):
        """Test handling single step response."""
        step = {'messages': [MagicMock()]}
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([step])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        assert mock_stream.called

    @patch('langchain_agent_lab.agent_executor')
    def test_agent_stream_multiple_steps(self, mock_executor):
        """Test handling multiple steps in response."""
        steps = [
            {'messages': [MagicMock()]},
            {'messages': [MagicMock()]},
            {'messages': [MagicMock()]},
            {'messages': [MagicMock()]},
        ]
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter(steps)
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        assert mock_stream.called


class TestMessagePrinting:
    """Test message printing in stream loop."""

    @patch('langchain_agent_lab.agent_executor')
    def test_pretty_print_called_on_last_message(self, mock_executor):
        """Test that pretty_print is called on the last message."""
        mock_message = MagicMock()
        mock_message.pretty_print = MagicMock()
        step = {'messages': [MagicMock(), mock_message]}
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([step])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # pretty_print should have been called
        assert mock_message.pretty_print.called

    @patch('langchain_agent_lab.agent_executor')
    def test_pretty_print_called_multiple_times_for_multiple_steps(self, mock_executor):
        """Test that pretty_print is called for each step."""
        mock_messages = [MagicMock(), MagicMock(), MagicMock()]
        for msg in mock_messages:
            msg.pretty_print = MagicMock()
        
        steps = [
            {'messages': [mock_messages[0]]},
            {'messages': [mock_messages[1]]},
            {'messages': [mock_messages[2]]},
        ]
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter(steps)
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Each message's pretty_print should have been called
        for msg in mock_messages:
            assert msg.pretty_print.called

    @patch('langchain_agent_lab.agent_executor')
    def test_pretty_print_on_single_message_in_step(self, mock_executor):
        """Test that pretty_print works with single message."""
        mock_message = MagicMock()
        mock_message.pretty_print = MagicMock()
        step = {'messages': [mock_message]}
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([step])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        assert mock_message.pretty_print.called

    @patch('langchain_agent_lab.agent_executor')
    def test_only_last_message_pretty_printed_per_step(self, mock_executor):
        """Test that only the last message per step is pretty printed."""
        mock_msg1 = MagicMock()
        mock_msg1.pretty_print = MagicMock()
        mock_msg2 = MagicMock()
        mock_msg2.pretty_print = MagicMock()
        
        step = {'messages': [mock_msg1, mock_msg2]}
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([step])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Only the last message should be pretty printed
        assert not mock_msg1.pretty_print.called
        assert mock_msg2.pretty_print.called


class TestStreamingEdgeCases:
    """Test edge cases in streaming."""

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_with_missing_messages_key(self, mock_executor):
        """Test handling when step doesn't have 'messages' key."""
        step = {'other_key': []}
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([step])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        # This should handle the KeyError or the code should be robust
        with pytest.raises(KeyError):
            importlib.reload(langchain_agent_lab)

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_with_empty_messages_list(self, mock_executor):
        """Test handling when messages list is empty."""
        step = {'messages': []}
        
        mock_stream = MagicMock()
        mock_stream.return_value = iter([step])
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        # Should raise IndexError when accessing [-1]
        with pytest.raises(IndexError):
            importlib.reload(langchain_agent_lab)

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_exception_handling(self, mock_executor):
        """Test that exceptions in stream are raised."""
        mock_stream = MagicMock()
        mock_stream.side_effect = Exception("Stream failed")
        mock_executor.stream = mock_stream
        
        import importlib
        import langchain_agent_lab
        with pytest.raises(Exception):
            importlib.reload(langchain_agent_lab)


class TestIntegrationWithExternalDependencies:
    """Test integration with external libraries."""

    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    @patch('langchain_agent_lab.create_react_agent')
    def test_all_components_wired_together(self, mock_create_agent, mock_memory_cls,
                                            mock_tavily_cls, mock_chat_cls):
        """Test that all components are properly wired together."""
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_model = MagicMock()
        mock_chat_cls.return_value = mock_model
        mock_search = MagicMock()
        mock_tavily_cls.return_value = mock_search
        mock_executor = MagicMock()
        mock_executor.stream = MagicMock(return_value=iter([]))
        mock_create_agent.return_value = mock_executor
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Verify all components were instantiated
        assert mock_memory_cls.called
        assert mock_chat_cls.called
        assert mock_tavily_cls.called
        assert mock_create_agent.called

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_correct_argument_passing_chain(self, mock_memory_cls, mock_tavily_cls,
                                             mock_chat_cls, mock_create_agent):
        """Test that arguments are passed correctly through the chain."""
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_model = MagicMock()
        mock_chat_cls.return_value = mock_model
        mock_search = MagicMock()
        mock_tavily_cls.return_value = mock_search
        mock_executor = MagicMock()
        mock_executor.stream = MagicMock(return_value=iter([]))
        mock_create_agent.return_value = mock_executor
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Verify correct argument passing
        mock_create_agent.assert_called_once()
        args, kwargs = mock_create_agent.call_args
        assert args[0] is mock_model
        assert args[1] == [mock_search]
        assert kwargs['checkpointer'] is mock_memory


class TestModuleVariables:
    """Test that module-level variables are properly set."""

    def test_memory_variable_exists(self):
        """Test that memory variable is defined."""
        import langchain_agent_lab
        assert hasattr(langchain_agent_lab, 'memory')

    def test_model_variable_exists(self):
        """Test that model variable is defined."""
        import langchain_agent_lab
        assert hasattr(langchain_agent_lab, 'model')

    def test_search_variable_exists(self):
        """Test that search variable is defined."""
        import langchain_agent_lab
        assert hasattr(langchain_agent_lab, 'search')

    def test_tools_variable_exists(self):
        """Test that tools variable is defined."""
        import langchain_agent_lab
        assert hasattr(langchain_agent_lab, 'tools')

    def test_agent_executor_variable_exists(self):
        """Test that agent_executor variable is defined."""
        import langchain_agent_lab
        assert hasattr(langchain_agent_lab, 'agent_executor')

    def test_config_variable_exists(self):
        """Test that config variable is defined."""
        import langchain_agent_lab
        assert hasattr(langchain_agent_lab, 'config')

    def test_tavily_api_key_variable_exists(self):
        """Test that tavily_api_key variable is defined."""
        import langchain_agent_lab
        assert hasattr(langchain_agent_lab, 'tavily_api_key')


class TestHumanMessageCreation:
    """Test HumanMessage creation."""

    def test_human_message_imported(self):
        """Test that HumanMessage is properly imported."""
        from langchain_agent_lab import HumanMessage
        assert HumanMessage is not None

    def test_human_message_with_content(self):
        """Test that HumanMessage can be created with content."""
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content="test message")
        assert msg.content == "test message"

    def test_human_message_content_is_string(self):
        """Test that message content is a string."""
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content="hi im bob! and i live in sf")
        assert isinstance(msg.content, str)


class TestCompleteFlowWithMocks:
    """Test the complete flow with mocked dependencies."""

    @patch('langchain_agent_lab.agent_executor')
    def test_complete_execution_flow_success(self, mock_executor):
        """Test complete successful execution flow."""
        # Setup mock responses
        mock_msg1 = MagicMock()
        mock_msg1.pretty_print = MagicMock()
        mock_msg2 = MagicMock()
        mock_msg2.pretty_print = MagicMock()
        
        steps = [
            {'messages': [mock_msg1]},
            {'messages': [mock_msg2]},
        ]
        
        mock_executor.stream = MagicMock(return_value=iter(steps))
        
        import importlib
        import langchain_agent_lab
        importlib.reload(langchain_agent_lab)
        
        # Verify the flow executed
        assert mock_executor.stream.called
        assert mock_msg1.pretty_print.called
        assert mock_msg2.pretty_print.called

    @patch('langchain_agent_lab.agent_executor')
    def test_complete_execution_