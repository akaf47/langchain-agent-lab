```python
import os
import pytest
from unittest.mock import patch, MagicMock, Mock, call
from dotenv import load_dotenv
from typing import Iterator, Dict, Any, List

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


class TestEnvironmentLoading:
    """Test environment variable loading functionality"""
    
    def test_load_dotenv_called(self):
        """Test that load_dotenv is called to load environment variables"""
        with patch('langchain_agent_lab.load_dotenv') as mock_load:
            # Re-import to trigger the load_dotenv call in module
            import importlib
            import langchain_agent_lab
            importlib.reload(langchain_agent_lab)
            mock_load.assert_called()
    
    def test_tavily_api_key_from_environment(self):
        """Test retrieving TAVILY_API_KEY from environment"""
        test_key = "test_tavily_key_12345"
        with patch.dict(os.environ, {'TAVILY_API_KEY': test_key}):
            retrieved_key = os.getenv("TAVILY_API_KEY")
            assert retrieved_key == test_key
    
    def test_tavily_api_key_not_in_environment(self):
        """Test retrieving TAVILY_API_KEY when not set"""
        with patch.dict(os.environ, {}, clear=True):
            retrieved_key = os.getenv("TAVILY_API_KEY")
            assert retrieved_key is None
    
    def test_tavily_api_key_default_value(self):
        """Test retrieving TAVILY_API_KEY with default value when not set"""
        with patch.dict(os.environ, {}, clear=True):
            retrieved_key = os.getenv("TAVILY_API_KEY", "default_key")
            assert retrieved_key == "default_key"
    
    def test_empty_string_tavily_api_key(self):
        """Test TAVILY_API_KEY as empty string"""
        with patch.dict(os.environ, {'TAVILY_API_KEY': ''}):
            retrieved_key = os.getenv("TAVILY_API_KEY")
            assert retrieved_key == ''


class TestMemorySaverInitialization:
    """Test MemorySaver component initialization"""
    
    def test_memory_saver_creation(self):
        """Test that MemorySaver is created successfully"""
        memory = MemorySaver()
        assert memory is not None
        assert isinstance(memory, MemorySaver)
    
    def test_memory_saver_is_singleton_like(self):
        """Test multiple MemorySaver instances can be created"""
        memory1 = MemorySaver()
        memory2 = MemorySaver()
        assert memory1 is not None
        assert memory2 is not None
        # They should be different instances
        assert memory1 is not memory2


class TestChatAnthropicModelInitialization:
    """Test ChatAnthropic model initialization"""
    
    def test_chat_anthropic_model_creation(self):
        """Test that ChatAnthropic model is created with correct model name"""
        model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
        assert model is not None
        assert model.model_name == "claude-3-sonnet-20240229"
    
    def test_chat_anthropic_model_name_exact(self):
        """Test exact model name assignment"""
        model_name = "claude-3-sonnet-20240229"
        model = ChatAnthropic(model_name=model_name)
        assert model.model_name == model_name
    
    def test_chat_anthropic_default_parameters(self):
        """Test ChatAnthropic initialization with default parameters"""
        model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
        # Verify basic properties
        assert hasattr(model, 'model_name')
        assert model.model_name is not None


class TestTavilySearchToolInitialization:
    """Test TavilySearchResults tool initialization"""
    
    def test_tavily_search_with_valid_api_key(self):
        """Test TavilySearchResults creation with valid API key"""
        api_key = "test_tavily_key_12345"
        search = TavilySearchResults(max_results=2, api_key=api_key)
        assert search is not None
        assert isinstance(search, TavilySearchResults)
        assert search.max_results == 2
    
    def test_tavily_search_max_results_parameter(self):
        """Test TavilySearchResults max_results parameter"""
        search = TavilySearchResults(max_results=2, api_key="test_key")
        assert search.max_results == 2
    
    def test_tavily_search_different_max_results(self):
        """Test TavilySearchResults with different max_results values"""
        for max_results_val in [1, 2, 5, 10]:
            search = TavilySearchResults(max_results=max_results_val, api_key="test_key")
            assert search.max_results == max_results_val
    
    def test_tavily_search_without_api_key_none(self):
        """Test TavilySearchResults creation when api_key is None"""
        with pytest.raises((ValueError, TypeError)):
            TavilySearchResults(max_results=2, api_key=None)
    
    def test_tavily_search_with_empty_api_key(self):
        """Test TavilySearchResults with empty string API key"""
        with pytest.raises((ValueError, TypeError)):
            TavilySearchResults(max_results=2, api_key='')


class TestToolsListConstruction:
    """Test construction of tools list"""
    
    def test_tools_list_contains_search_tool(self):
        """Test that tools list contains TavilySearchResults"""
        search = TavilySearchResults(max_results=2, api_key="test_key")
        tools = [search]
        assert len(tools) == 1
        assert tools[0] is search
    
    def test_tools_list_is_mutable(self):
        """Test that tools list can be manipulated"""
        search1 = TavilySearchResults(max_results=2, api_key="test_key_1")
        search2 = TavilySearchResults(max_results=3, api_key="test_key_2")
        tools = [search1]
        assert len(tools) == 1
        tools.append(search2)
        assert len(tools) == 2
        assert tools[1] is search2


class TestAgentExecutorCreation:
    """Test agent executor creation"""
    
    def test_agent_executor_created_successfully(self):
        """Test that agent executor is created without errors"""
        memory = MemorySaver()
        model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
        search = TavilySearchResults(max_results=2, api_key="test_key")
        tools = [search]
        
        agent_executor = create_react_agent(model, tools, checkpointer=memory)
        assert agent_executor is not None
    
    def test_agent_executor_with_empty_tools_list(self):
        """Test agent executor creation with empty tools list"""
        memory = MemorySaver()
        model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
        tools = []
        
        agent_executor = create_react_agent(model, tools, checkpointer=memory)
        assert agent_executor is not None
    
    def test_agent_executor_with_memory_checkpointer(self):
        """Test that memory checkpointer is passed correctly"""
        memory = MemorySaver()
        model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
        search = TavilySearchResults(max_results=2, api_key="test_key")
        tools = [search]
        
        # The checkpointer should be passed through
        agent_executor = create_react_agent(model, tools, checkpointer=memory)
        assert agent_executor is not None


class TestConfigurationSetup:
    """Test configuration dictionary setup"""
    
    def test_config_dict_creation(self):
        """Test that config dictionary is created with thread_id"""
        config = {"configurable": {"thread_id": "abc123"}}
        assert config is not None
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == "abc123"
    
    def test_config_thread_id_value(self):
        """Test thread_id value in config"""
        thread_id = "abc123"
        config = {"configurable": {"thread_id": thread_id}}
        assert config["configurable"]["thread_id"] == thread_id
    
    def test_config_with_different_thread_id(self):
        """Test config with different thread_id values"""
        thread_ids = ["thread_1", "abc123", "xyz789", "test_thread"]
        for thread_id in thread_ids:
            config = {"configurable": {"thread_id": thread_id}}
            assert config["configurable"]["thread_id"] == thread_id
    
    def test_config_empty_thread_id(self):
        """Test config with empty thread_id"""
        config = {"configurable": {"thread_id": ""}}
        assert config["configurable"]["thread_id"] == ""
    
    def test_config_nested_structure(self):
        """Test config nested dictionary structure"""
        config = {"configurable": {"thread_id": "abc123"}}
        assert isinstance(config, dict)
        assert isinstance(config["configurable"], dict)
        assert isinstance(config["configurable"]["thread_id"], str)


class TestHumanMessageCreation:
    """Test HumanMessage creation and usage"""
    
    def test_human_message_creation(self):
        """Test creating HumanMessage with content"""
        message = HumanMessage(content="hi im bob! and i live in sf")
        assert message is not None
        assert isinstance(message, HumanMessage)
        assert message.content == "hi im bob! and i live in sf"
    
    def test_human_message_content_exact(self):
        """Test exact content in HumanMessage"""
        content = "hi im bob! and i live in sf"
        message = HumanMessage(content=content)
        assert message.content == content
    
    def test_human_message_with_empty_string(self):
        """Test HumanMessage with empty string content"""
        message = HumanMessage(content="")
        assert message.content == ""
    
    def test_human_message_with_special_characters(self):
        """Test HumanMessage with special characters"""
        content = "Test @#$%^&*()_+-=[]{}|;:',.<>?/`~"
        message = HumanMessage(content=content)
        assert message.content == content
    
    def test_human_message_with_unicode(self):
        """Test HumanMessage with unicode characters"""
        content = "Hello 世界 🌍 Привет"
        message = HumanMessage(content=content)
        assert message.content == content
    
    def test_human_message_wrapping_in_list(self):
        """Test wrapping HumanMessage in a list"""
        message = HumanMessage(content="test")
        messages = [message]
        assert len(messages) == 1
        assert messages[0] is message


class TestAgentStreamExecution:
    """Test agent stream execution"""
    
    def test_agent_stream_basic_execution(self):
        """Test basic agent stream execution"""
        with patch('langchain_agent_lab.create_react_agent') as mock_create:
            mock_agent = MagicMock()
            mock_agent.stream = MagicMock(return_value=iter([
                {"messages": [HumanMessage(content="test")]}
            ]))
            mock_create.return_value = mock_agent
            
            config = {"configurable": {"thread_id": "abc123"}}
            input_data = {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}
            
            # Simulate stream call
            steps = list(mock_agent.stream(input_data, config, stream_mode="values"))
            
            assert len(steps) > 0
            assert "messages" in steps[0]
    
    def test_agent_stream_with_config(self):
        """Test agent stream with configuration"""
        mock_agent = MagicMock()
        test_response = {"messages": [HumanMessage(content="response")]}
        mock_agent.stream = MagicMock(return_value=iter([test_response]))
        
        config = {"configurable": {"thread_id": "abc123"}}
        input_data = {"messages": [HumanMessage(content="test")]}
        
        # Call stream
        steps = list(mock_agent.stream(input_data, config, stream_mode="values"))
        
        # Verify stream was called with correct arguments
        mock_agent.stream.assert_called_once_with(input_data, config, stream_mode="values")
    
    def test_agent_stream_returns_iterator(self):
        """Test that stream returns an iterator"""
        mock_agent = MagicMock()
        mock_agent.stream = MagicMock(return_value=iter([
            {"messages": [HumanMessage(content="msg1")]},
            {"messages": [HumanMessage(content="msg2")]}
        ]))
        
        result = mock_agent.stream({"messages": []}, {}, stream_mode="values")
        
        # Convert to list to consume iterator
        steps = list(result)
        assert len(steps) == 2
    
    def test_agent_stream_multiple_messages(self):
        """Test stream with multiple message iterations"""
        mock_agent = MagicMock()
        messages_sequence = [
            {"messages": [HumanMessage(content=f"msg_{i}")]},
            {"messages": [HumanMessage(content=f"response_{i}")]}
            for i in range(3)
        ]
        mock_agent.stream = MagicMock(return_value=iter(messages_sequence))
        
        steps = list(mock_agent.stream({"messages": []}, {}, stream_mode="values"))
        assert len(steps) == 6


class TestStreamModeParameter:
    """Test stream_mode parameter"""
    
    def test_stream_mode_values(self):
        """Test stream_mode='values' parameter"""
        mock_agent = MagicMock()
        mock_agent.stream = MagicMock(return_value=iter([]))
        
        config = {"configurable": {"thread_id": "test"}}
        input_data = {"messages": []}
        
        mock_agent.stream(input_data, config, stream_mode="values")
        
        # Verify stream_mode parameter was passed
        _, kwargs = mock_agent.stream.call_args
        assert kwargs.get('stream_mode') == "values"
    
    def test_stream_mode_updates(self):
        """Test stream_mode='updates' parameter"""
        mock_agent = MagicMock()
        mock_agent.stream = MagicMock(return_value=iter([]))
        
        config = {"configurable": {"thread_id": "test"}}
        input_data = {"messages": []}
        
        mock_agent.stream(input_data, config, stream_mode="updates")
        
        _, kwargs = mock_agent.stream.call_args
        assert kwargs.get('stream_mode') == "updates"


class TestMessagePrettyPrint:
    """Test message pretty_print method"""
    
    def test_message_pretty_print_callable(self):
        """Test that message has pretty_print method"""
        message = HumanMessage(content="test")
        assert hasattr(message, 'pretty_print')
        assert callable(message.pretty_print)
    
    def test_message_pretty_print_execution(self):
        """Test executing pretty_print method"""
        message = HumanMessage(content="test content")
        # Should not raise an error
        with patch.object(message, 'pretty_print') as mock_print:
            message.pretty_print()
            mock_print.assert_called_once()
    
    def test_ai_message_pretty_print(self):
        """Test pretty_print on AI message"""
        message = AIMessage(content="AI response")
        assert hasattr(message, 'pretty_print')
        with patch.object(message, 'pretty_print') as mock_print:
            message.pretty_print()
            mock_print.assert_called_once()


class TestLastMessageAccess:
    """Test accessing last message in messages list"""
    
    def test_access_last_message_single(self):
        """Test accessing last message when only one message exists"""
        messages = [HumanMessage(content="test")]
        last_message = messages[-1]
        assert last_message.content == "test"
    
    def test_access_last_message_multiple(self):
        """Test accessing last message with multiple messages"""
        messages = [
            HumanMessage(content="first"),
            HumanMessage(content="second"),
            HumanMessage(content="third")
        ]
        last_message = messages[-1]
        assert last_message.content == "third"
    
    def test_access_last_message_in_dict(self):
        """Test accessing last message from dict structure"""
        step = {"messages": [
            HumanMessage(content="msg1"),
            HumanMessage(content="msg2")
        ]}
        last_message = step["messages"][-1]
        assert last_message.content == "msg2"
    
    def test_last_message_index_negative_one(self):
        """Test that -1 index accesses last element"""
        messages = [HumanMessage(content="only")]
        assert messages[-1] is messages[0]


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components"""
    
    def test_full_agent_setup_chain(self):
        """Test complete agent setup chain"""
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'}):
            # Load environment
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            assert tavily_api_key == 'test_key'
            
            # Create components
            memory = MemorySaver()
            model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
            search = TavilySearchResults(max_results=2, api_key=tavily_api_key)
            tools = [search]
            
            # Create agent
            agent_executor = create_react_agent(model, tools, checkpointer=memory)
            assert agent_executor is not None
    
    def test_stream_execution_with_real_structures(self):
        """Test stream execution with realistic data structures"""
        with patch('langchain_agent_lab.create_react_agent') as mock_create:
            mock_agent = MagicMock()
            
            # Simulate agent stream output
            def stream_generator(*args, **kwargs):
                yield {
                    "messages": [
                        HumanMessage(content="hi im bob! and i live in sf"),
                        AIMessage(content="Nice to meet you, Bob!")
                    ]
                }
                yield {
                    "messages": [
                        HumanMessage(content="hi im bob! and i live in sf"),
                        AIMessage(content="San Francisco is a great place!"),
                        AIMessage(content="[Final response]")
                    ]
                }
            
            mock_agent.stream = MagicMock(side_effect=stream_generator)
            mock_create.return_value = mock_agent
            
            config = {"configurable": {"thread_id": "abc123"}}
            input_msg = HumanMessage(content="hi im bob! and i live in sf")
            
            steps = list(mock_agent.stream(
                {"messages": [input_msg]},
                config,
                stream_mode="values"
            ))
            
            assert len(steps) == 2
            assert len(steps[-1]["messages"]) >= 2
    
    def test_message_dict_access_flow(self):
        """Test the complete flow of accessing message from step dict"""
        # Simulate what happens in the actual code
        step = {"messages": [
            HumanMessage(content="initial"),
            AIMessage(content="response")
        ]}
        
        # Access last message (like step["messages"][-1])
        last_message = step["messages"][-1]
        
        # Should have pretty_print method
        assert hasattr(last_message, 'pretty_print')
        assert last_message.content == "response"


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions"""
    
    def test_very_long_message_content(self):
        """Test with very long message content"""
        long_content = "a" * 10000
        message = HumanMessage(content=long_content)
        assert len(message.content) == 10000
    
    def test_message_with_newlines(self):
        """Test message content with newlines"""
        content = "line1\nline2\nline3"
        message = HumanMessage(content=content)
        assert "\n" in message.content
        assert message.content.count("\n") == 2
    
    def test_message_with_tabs(self):
        """Test message content with tabs"""
        content = "col1\tcol2\tcol3"
        message = HumanMessage(content=content)
        assert "\t" in message.content
    
    def test_config_with_many_thread_ids(self):
        """Test creating many configs with different thread IDs"""
        configs = [
            {"configurable": {"thread_id": f"thread_{i}"}}
            for i in range(100)
        ]
        assert len(configs) == 100
        assert configs[0]["configurable"]["thread_id"] == "thread_0"
        assert configs[99]["configurable"]["thread_id"] == "thread_99"
    
    def test_tools_list_with_none_elements(self):
        """Test tools list behavior"""
        search = TavilySearchResults(max_results=2, api_key="test_key")
        tools = [search, search]  # Same tool twice
        assert len(tools) == 2
        assert tools[0] is tools[1]


class TestModuleLevel:
    """Test module-level code execution"""
    
    def test_module_imports_all_required_modules(self):
        """Test that all required modules are imported"""
        import langchain_agent_lab
        
        # Check key imports are available
        assert hasattr(langchain_agent_lab, 'ChatAnthropic')
        assert hasattr(langchain_agent_lab, 'TavilySearchResults')
        assert hasattr(langchain_agent_lab, 'HumanMessage')
        assert hasattr(langchain_agent_lab, 'MemorySaver')
        assert hasattr(langchain_agent_lab, 'create_react_agent')
    
    def test_module_creates_agent_executor(self):
        """Test that module-level code creates agent_executor"""
        import langchain_agent_lab
        
        # The module should have agent_executor at module level
        assert hasattr(langchain_agent_lab, 'agent_executor')


class TestErrorHandling:
    """Test error handling in various scenarios"""
    
    def test_chat_anthropic_with_none_model_name(self):
        """Test ChatAnthropic with None as model_name"""
        with pytest.raises((ValueError, TypeError)):
            ChatAnthropic(model_name=None)
    
    def test_memory_saver_basic_functionality(self):
        """Test MemorySaver basic functionality"""
        memory = MemorySaver()
        # Should be instantiable without errors
        assert memory is not None
    
    def test_create_react_agent_with_none_model(self):
        """Test create_react_agent with None model"""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            create_react_agent(None, [], checkpointer=MemorySaver())
    
    def test_create_react_agent_with_none_tools(self):
        """Test create_react_agent with None tools"""
        model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
        # Passing None for tools should either work or raise appropriate error
        try:
            agent = create_react_agent(model, None, checkpointer=MemorySaver())
            # If it works, that's valid
            assert agent is not None
        except (ValueError, TypeError, AttributeError):
            # If it raises, that's also valid error handling
            pass


class TestValueTypes:
    """Test various value types and their handling"""
    
    def test_thread_id_string_type(self):
        """Test thread_id must be string or stringifiable"""
        config = {"configurable": {"thread_id": "abc123"}}
        assert isinstance(config["configurable"]["thread_id"], str)
    
    def test_max_results_integer_type(self):
        """Test max_results is integer type"""
        search = TavilySearchResults(max_results=2, api_key="key")
        assert isinstance(search.max_results, int)
        assert search.max_results > 0
    
    def test_message_content_string_type(self):
        """Test message content is string type"""
        message = HumanMessage(content="test")
        assert isinstance(message.content, str)


class TestSequentialExecution:
    """Test sequential execution patterns"""
    
    def test_stream_step_by_step(self):
        """Test processing stream steps one by one"""
        mock_agent = MagicMock()
        steps_data = [
            {"messages": [HumanMessage(content=f"step_{i}")]},
            {"messages": [HumanMessage(content=f"response_{i}")]}
            for i in range(2)
        ]
        mock_agent.stream = MagicMock(return_value=iter(steps_data))
        
        for idx, step in enumerate(mock_agent.stream({}, {}, stream_mode="values")):
            assert "messages" in step
            assert len(step["messages"]) > 0
            if idx < 4:  # We created 4 steps
                assert step["messages"][0] is not None


class TestCompleteWorkflow:
    """Test complete workflow from start to finish"""
    
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'})
    @patch('langchain_agent_lab.create_react_agent')
    def test_complete_workflow(self, mock_create_agent):
        """Test complete agent workflow"""
        # Setup mock agent
        mock_agent = MagicMock()
        mock_agent.stream = MagicMock(return_value=iter([
            {
                "messages": [
                    HumanMessage(content="hi im bob! and i live in sf"),
                    AIMessage(content="Hello Bob from SF!")
                ]
            }
        ]))
        mock_create_agent.return_value = mock_agent
        
        # Create config
        config = {"configurable": {"thread_id": "abc123"}}
        
        # Create input message
        input_message = HumanMessage(content="hi im bob! and i live in sf")
        
        # Execute stream
        for step in mock_agent.stream(
            {"messages": [input_message]},
            config,
            stream_mode="values"
        ):
            # Simulate the pretty_print call
            last_msg = step["messages"][-1]
            assert last_msg is not None
```