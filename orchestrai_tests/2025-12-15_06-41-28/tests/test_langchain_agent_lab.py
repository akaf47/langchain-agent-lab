```python
import os
import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import the module to test
import langchain_agent_lab

class TestAgentInitialization:
    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly"""
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'}):
            load_dotenv()
            assert os.getenv("TAVILY_API_KEY") == 'test_key'

    def test_memory_saver_creation(self):
        """Test MemorySaver initialization"""
        memory = MemorySaver()
        assert memory is not None

    def test_anthropic_model_creation(self):
        """Test ChatAnthropic model initialization"""
        model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
        assert model is not None
        assert model.model_name == "claude-3-sonnet-20240229"

    def test_tavily_search_tool_creation(self):
        """Test TavilySearchResults tool initialization"""
        with patch.dict(os.environ, {'TAVILY_API_KEY': 'test_key'}):
            search = TavilySearchResults(max_results=2, api_key='test_key')
            assert search is not None
            assert search.max_results == 2

    def test_agent_executor_creation(self):
        """Test agent executor creation"""
        model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
        search = TavilySearchResults(max_results=2, api_key='test_api_key')
        tools = [search]
        
        agent_executor = create_react_agent(model, tools, checkpointer=MemorySaver())
        assert agent_executor is not None

class TestAgentExecution:
    @pytest.fixture
    def mock_agent_executor(self):
        """Create a mock agent executor for testing"""
        mock_model = MagicMock(spec=ChatAnthropic)
        mock_search = MagicMock(spec=TavilySearchResults)
        mock_memory = MagicMock(spec=MemorySaver)
        
        return create_react_agent(mock_model, [mock_search], checkpointer=mock_memory)

    def test_agent_stream_configuration(self, mock_agent_executor):
        """Test agent stream configuration"""
        config = {"configurable": {"thread_id": "abc123"}}
        
        # Mock the stream method to return a controlled response
        mock_agent_executor.stream = MagicMock(return_value=[
            {"messages": [HumanMessage(content="test response")]}
        ])
        
        steps = list(mock_agent_executor.stream(
            {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
            config,
            stream_mode="values"
        ))
        
        assert len(steps) > 0
        assert steps[0]["messages"][-1].content is not None

    def test_agent_execution_without_api_key(self):
        """Test agent execution behavior when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                TavilySearchResults(max_results=2)

    def test_message_processing(self, mock_agent_executor):
        """Test message processing through the agent"""
        input_message = HumanMessage(content="Test message")
        config = {"configurable": {"thread_id": "test_thread"}}
        
        mock_agent_executor.stream = MagicMock(return_value=[
            {"messages": [input_message, HumanMessage(content="Agent response")]}
        ])
        
        steps = list(mock_agent_executor.stream(
            {"messages": [input_message]},
            config,
            stream_mode="values"
        ))
        
        assert len(steps) > 0
        assert len(steps[0]["messages"]) > 1
```