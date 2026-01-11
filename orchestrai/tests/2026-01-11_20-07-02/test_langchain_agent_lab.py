"""
Comprehensive test suite for langchain_agent_lab.py
Tests cover all code paths, error scenarios, and edge cases.
Target: 100% code coverage
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch, call, PropertyMock
from typing import Dict, List, Any
from importlib import reload
import importlib


class TestEnvironmentVariableLoading:
    """Test environment variable loading and .env file handling."""

    def test_load_dotenv_is_called_on_module_import(self):
        """Should call load_dotenv when module is imported."""
        with patch('langchain_agent_lab.load_dotenv') as mock_load:
            # Force reload to trigger module-level code
            if 'langchain_agent_lab' in sys.modules:
                del sys.modules['langchain_agent_lab']
            import langchain_agent_lab
            mock_load.assert_called_once()

    def test_tavily_api_key_retrieved_from_environment(self):
        """Should retrieve TAVILY_API_KEY from environment variables."""
        test_key = "sk_test_tavily_12345"
        with patch.dict(os.environ, {'TAVILY_API_KEY': test_key}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                assert langchain_agent_lab.tavily_api_key == test_key

    def test_tavily_api_key_none_when_not_in_environment(self):
        """Should return None when TAVILY_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                assert langchain_agent_lab.tavily_api_key is None

    def test_tavily_api_key_empty_string_when_set_to_empty(self):
        """Should return empty string when API key is set to empty."""
        with patch.dict(os.environ, {'TAVILY_API_KEY': ''}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                assert langchain_agent_lab.tavily_api_key == ''

    def test_tavily_api_key_with_special_characters(self):
        """Should handle API keys with special characters."""
        special_key = "sk-test!@#$%^&*()_+-=[]{}|;:,.<>?"
        with patch.dict(os.environ, {'TAVILY_API_KEY': special_key}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                assert langchain_agent_lab.tavily_api_key == special_key


class TestMemorySaverInitialization:
    """Test MemorySaver component initialization."""

    @patch('langchain_agent_lab.MemorySaver')
    def test_memory_saver_is_instantiated(self, mock_memory_class):
        """Should create MemorySaver instance."""
        mock_instance = MagicMock()
        mock_memory_class.return_value = mock_instance
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        mock_memory_class.assert_called_once()
        assert langchain_agent_lab.memory == mock_instance

    @patch('langchain_agent_lab.MemorySaver')
    def test_memory_saver_called_with_no_arguments(self, mock_memory_class):
        """Should create MemorySaver with default initialization."""
        mock_instance = MagicMock()
        mock_memory_class.return_value = mock_instance
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        # Should be called with no arguments
        mock_memory_class.assert_called_once_with()

    @patch('langchain_agent_lab.MemorySaver')
    def test_memory_assigned_to_module_variable(self, mock_memory_class):
        """Should assign MemorySaver instance to memory variable."""
        mock_instance = MagicMock()
        mock_memory_class.return_value = mock_instance
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        assert hasattr(langchain_agent_lab, 'memory')
        assert langchain_agent_lab.memory is mock_instance


class TestChatAnthropicModelInitialization:
    """Test ChatAnthropic model initialization."""

    @patch('langchain_agent_lab.ChatAnthropic')
    def test_chat_anthropic_is_instantiated(self, mock_chat_class):
        """Should create ChatAnthropic instance."""
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        mock_chat_class.assert_called_once()
        assert langchain_agent_lab.model == mock_instance

    @patch('langchain_agent_lab.ChatAnthropic')
    def test_chat_anthropic_uses_claude_3_sonnet(self, mock_chat_class):
        """Should use claude-3-sonnet-20240229 model."""
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs['model_name'] == "claude-3-sonnet-20240229"

    @patch('langchain_agent_lab.ChatAnthropic')
    def test_chat_anthropic_model_name_parameter(self, mock_chat_class):
        """Should pass model_name as keyword argument."""
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        args, kwargs = mock_chat_class.call_args
        assert 'model_name' in kwargs
        assert kwargs['model_name'] == "claude-3-sonnet-20240229"

    @patch('langchain_agent_lab.ChatAnthropic')
    def test_model_assigned_to_module_variable(self, mock_chat_class):
        """Should assign ChatAnthropic instance to model variable."""
        mock_instance = MagicMock()
        mock_chat_class.return_value = mock_instance
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        assert hasattr(langchain_agent_lab, 'model')
        assert langchain_agent_lab.model is mock_instance


class TestTavilySearchToolInitialization:
    """Test TavilySearchResults tool initialization."""

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tavily_search_results_instantiated(self, mock_tavily_class):
        """Should create TavilySearchResults instance."""
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                mock_tavily_class.assert_called_once()
                assert langchain_agent_lab.search == mock_instance

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tavily_search_max_results_is_two(self, mock_tavily_class):
        """Should set max_results to 2."""
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                call_kwargs = mock_tavily_class.call_args[1]
                assert call_kwargs['max_results'] == 2

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tavily_search_receives_api_key(self, mock_tavily_class):
        """Should pass tavily_api_key to TavilySearchResults."""
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        test_key = "tavily_test_key_xyz"
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': test_key}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                call_kwargs = mock_tavily_class.call_args[1]
                assert call_kwargs['api_key'] == test_key

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tavily_search_with_none_api_key(self, mock_tavily_class):
        """Should pass None API key when not set."""
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                call_kwargs = mock_tavily_class.call_args[1]
                assert call_kwargs['api_key'] is None

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_search_assigned_to_module_variable(self, mock_tavily_class):
        """Should assign TavilySearchResults instance to search variable."""
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                assert hasattr(langchain_agent_lab, 'search')
                assert langchain_agent_lab.search is mock_instance

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tavily_search_kwargs_max_results_and_api_key(self, mock_tavily_class):
        """Should pass both max_results and api_key as kwargs."""
        mock_instance = MagicMock()
        mock_tavily_class.return_value = mock_instance
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'key123'}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                call_kwargs = mock_tavily_class.call_args[1]
                assert 'max_results' in call_kwargs
                assert 'api_key' in call_kwargs
                assert call_kwargs['max_results'] == 2
                assert call_kwargs['api_key'] == 'key123'


class TestToolsListConfiguration:
    """Test tools list creation and configuration."""

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tools_list_is_created(self, mock_tavily_class):
        """Should create tools list."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test'}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                assert hasattr(langchain_agent_lab, 'tools')
                assert isinstance(langchain_agent_lab.tools, list)

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tools_list_contains_search_tool(self, mock_tavily_class):
        """Should include search tool in tools list."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test'}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                assert mock_search in langchain_agent_lab.tools

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tools_list_has_exactly_one_element(self, mock_tavily_class):
        """Should have exactly one tool in the list."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test'}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                assert len(langchain_agent_lab.tools) == 1

    @patch('langchain_agent_lab.TavilySearchResults')
    def test_tools_list_first_element_is_search(self, mock_tavily_class):
        """Should have search as the first (and only) element."""
        mock_search = MagicMock()
        mock_tavily_class.return_value = mock_search
        
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test'}, clear=True):
            with patch('langchain_agent_lab.load_dotenv'):
                if 'langchain_agent_lab' in sys.modules:
                    del sys.modules['langchain_agent_lab']
                import langchain_agent_lab
                
                assert langchain_agent_lab.tools[0] is mock_search


class TestCreateReactAgentCall:
    """Test create_react_agent function call."""

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_create_react_agent_is_called(self, mock_mem_cls, mock_tav_cls,
                                          mock_chat_cls, mock_create_agent):
        """Should call create_react_agent."""
        mock_mem = MagicMock()
        mock_mem_cls.return_value = mock_mem
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        mock_search = MagicMock()
        mock_tav_cls.return_value = mock_search
        mock_executor = MagicMock()
        mock_executor.stream = MagicMock(return_value=iter([]))
        mock_create_agent.return_value = mock_executor
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        mock_create_agent.assert_called_once()

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_create_react_agent_receives_model(self, mock_mem_cls, mock_tav_cls,
                                               mock_chat_cls, mock_create_agent):
        """Should pass model as first positional argument."""
        mock_mem = MagicMock()
        mock_mem_cls.return_value = mock_mem
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        mock_search = MagicMock()
        mock_tav_cls.return_value = mock_search
        mock_executor = MagicMock()
        mock_executor.stream = MagicMock(return_value=iter([]))
        mock_create_agent.return_value = mock_executor
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        args, kwargs = mock_create_agent.call_args
        assert args[0] is mock_chat

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_create_react_agent_receives_tools(self, mock_mem_cls, mock_tav_cls,
                                               mock_chat_cls, mock_create_agent):
        """Should pass tools list as second positional argument."""
        mock_mem = MagicMock()
        mock_mem_cls.return_value = mock_mem
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        mock_search = MagicMock()
        mock_tav_cls.return_value = mock_search
        mock_executor = MagicMock()
        mock_executor.stream = MagicMock(return_value=iter([]))
        mock_create_agent.return_value = mock_executor
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        args, kwargs = mock_create_agent.call_args
        assert args[1] == [mock_search]

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_create_react_agent_receives_checkpointer_kwarg(self, mock_mem_cls,
                                                           mock_tav_cls,
                                                           mock_chat_cls,
                                                           mock_create_agent):
        """Should pass memory as checkpointer keyword argument."""
        mock_mem = MagicMock()
        mock_mem_cls.return_value = mock_mem
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        mock_search = MagicMock()
        mock_tav_cls.return_value = mock_search
        mock_executor = MagicMock()
        mock_executor.stream = MagicMock(return_value=iter([]))
        mock_create_agent.return_value = mock_executor
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        args, kwargs = mock_create_agent.call_args
        assert 'checkpointer' in kwargs
        assert kwargs['checkpointer'] is mock_mem

    @patch('langchain_agent_lab.create_react_agent')
    @patch('langchain_agent_lab.ChatAnthropic')
    @patch('langchain_agent_lab.TavilySearchResults')
    @patch('langchain_agent_lab.MemorySaver')
    def test_agent_executor_assigned_to_module_variable(self, mock_mem_cls,
                                                        mock_tav_cls,
                                                        mock_chat_cls,
                                                        mock_create_agent):
        """Should assign create_react_agent return value to agent_executor."""
        mock_mem = MagicMock()
        mock_mem_cls.return_value = mock_mem
        mock_chat = MagicMock()
        mock_chat_cls.return_value = mock_chat
        mock_search = MagicMock()
        mock_tav_cls.return_value = mock_search
        mock_executor = MagicMock()
        mock_executor.stream = MagicMock(return_value=iter([]))
        mock_create_agent.return_value = mock_executor
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        assert hasattr(langchain_agent_lab, 'agent_executor')
        assert langchain_agent_lab.agent_executor is mock_executor


class TestConfigurationDictionary:
    """Test agent configuration dictionary."""

    def test_config_variable_exists(self):
        """Should create config variable."""
        import langchain_agent_lab
        assert hasattr(langchain_agent_lab, 'config')

    def test_config_is_dictionary(self):
        """Should create config as a dict."""
        import langchain_agent_lab
        assert isinstance(langchain_agent_lab.config, dict)

    def test_config_has_configurable_key(self):
        """Should have 'configurable' key in config."""
        import langchain_agent_lab
        assert 'configurable' in langchain_agent_lab.config

    def test_config_configurable_is_dict(self):
        """Should have configurable as a dict."""
        import langchain_agent_lab
        assert isinstance(langchain_agent_lab.config['configurable'], dict)

    def test_config_has_thread_id(self):
        """Should have 'thread_id' in configurable."""
        import langchain_agent_lab
        assert 'thread_id' in langchain_agent_lab.config['configurable']

    def test_config_thread_id_is_abc123(self):
        """Should set thread_id to 'abc123'."""
        import langchain_agent_lab
        assert langchain_agent_lab.config['configurable']['thread_id'] == "abc123"

    def test_config_structure_complete(self):
        """Should have complete expected structure."""
        import langchain_agent_lab
        expected = {"configurable": {"thread_id": "abc123"}}
        assert langchain_agent_lab.config == expected

    def test_config_string_values(self):
        """Should have string values for thread_id."""
        import langchain_agent_lab
        thread_id = langchain_agent_lab.config['configurable']['thread_id']
        assert isinstance(thread_id, str)


class TestAgentStreamExecution:
    """Test agent.stream() execution."""

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_method_is_called(self, mock_executor):
        """Should call stream method on agent_executor."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        assert mock_executor.stream.called

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_receives_input_dict(self, mock_executor):
        """Should pass input dict as first argument to stream."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_args = mock_executor.stream.call_args
        input_dict = call_args[0][0]
        assert isinstance(input_dict, dict)
        assert 'messages' in input_dict

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_input_dict_has_messages_key(self, mock_executor):
        """Should include 'messages' key in input dict."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_args = mock_executor.stream.call_args
        input_dict = call_args[0][0]
        assert 'messages' in input_dict

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_messages_is_list(self, mock_executor):
        """Should have messages as a list."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_args = mock_executor.stream.call_args
        messages = call_args[0][0]['messages']
        assert isinstance(messages, list)

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_messages_has_one_element(self, mock_executor):
        """Should have exactly one message in the list."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_args = mock_executor.stream.call_args
        messages = call_args[0][0]['messages']
        assert len(messages) == 1

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_message_is_human_message(self, mock_executor):
        """Should pass HumanMessage instance."""
        from langchain_core.messages import HumanMessage
        
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_args = mock_executor.stream.call_args
        message = call_args[0][0]['messages'][0]
        assert isinstance(message, HumanMessage)

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_message_content(self, mock_executor):
        """Should pass correct message content."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_args = mock_executor.stream.call_args
        message = call_args[0][0]['messages'][0]
        assert message.content == "hi im bob! and i live in sf"

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_receives_config_as_second_argument(self, mock_executor):
        """Should pass config as second positional argument."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_args = mock_executor.stream.call_args
        config = call_args[0][1]
        assert config == {"configurable": {"thread_id": "abc123"}}

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_receives_stream_mode_kwarg(self, mock_executor):
        """Should pass stream_mode as keyword argument."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_kwargs = mock_executor.stream.call_args[1]
        assert 'stream_mode' in call_kwargs

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_mode_is_values(self, mock_executor):
        """Should set stream_mode to 'values'."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        call_kwargs = mock_executor.stream.call_args[1]
        assert call_kwargs['stream_mode'] == 'values'


class TestStreamIterationLoop:
    """Test the for loop iterating over stream results."""

    @patch('langchain_agent_lab.agent_executor')
    def test_stream_returns_iterable(self, mock_executor):
        """Should iterate over stream results."""
        mock_executor.stream = MagicMock(return_value=iter([]))
        
        if 'langchain_agent_lab' in sys.modules:
            del sys.modules['langchain_agent_lab']
        import langchain_agent_lab
        
        # Should execute without error
        assert mock_executor.stream.called

    @patch('langchain_agent_lab.agent_executor')
    def test_iteration_with_empty_stream