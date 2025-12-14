import importlib
import sys
import types
import os
import pytest


def setup_fake_modules(monkeypatch):
    """
    Create and register fake modules/classes/functions required by langchain_agent_lab.py
    Returns a container dict collecting all interactions for assertions.
    """
    container = {
        "dotenv_called": 0,
        "chatanthropic_inits": [],
        "tavily_inits": [],
        "memory_inits": [],
        "create_agent_calls": [],
        "stream_calls": [],
        "pretty_print_calls": 0,
        "model_instances": [],
        "search_instances": [],
        "memory_instances": [],
        "humanmessage_class": None,
    }

    # Fake dotenv
    dotenv_mod = types.ModuleType("dotenv")

    def fake_load_dotenv(*args, **kwargs):
        container["dotenv_called"] += 1

    dotenv_mod.load_dotenv = fake_load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_mod)

    # Fake langchain_core.messages with HumanMessage
    lc_core_mod = types.ModuleType("langchain_core")
    lc_core_messages_mod = types.ModuleType("langchain_core.messages")

    class FakeHumanMessage:
        def __init__(self, content=None, **kwargs):
            self.content = content
            self.kwargs = kwargs

    lc_core_messages_mod.HumanMessage = FakeHumanMessage
    container["humanmessage_class"] = FakeHumanMessage
    monkeypatch.setitem(sys.modules, "langchain_core", lc_core_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", lc_core_messages_mod)

    # Fake langchain_anthropic with ChatAnthropic
    lc_anthropic_mod = types.ModuleType("langchain_anthropic")

    class FakeChatAnthropic:
        def __init__(self, *args, **kwargs):
            # store args/kwargs for verification
            self.args = args
            self.kwargs = kwargs
            container["chatanthropic_inits"].append({"args": args, "kwargs": kwargs})
            container["model_instances"].append(self)

    lc_anthropic_mod.ChatAnthropic = FakeChatAnthropic
    monkeypatch.setitem(sys.modules, "langchain_anthropic", lc_anthropic_mod)

    # Fake langchain_community.tools.tavily_search with TavilySearchResults
    lc_comm_mod = types.ModuleType("langchain_community")
    lc_comm_tools_mod = types.ModuleType("langchain_community.tools")
    lc_comm_tavily_mod = types.ModuleType("langchain_community.tools.tavily_search")

    class FakeTavilySearchResults:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            container["tavily_inits"].append({"args": args, "kwargs": kwargs})
            container["search_instances"].append(self)

    lc_comm_tavily_mod.TavilySearchResults = FakeTavilySearchResults
    monkeypatch.setitem(sys.modules, "langchain_community", lc_comm_mod)
    monkeypatch.setitem(sys.modules, "langchain_community.tools", lc_comm_tools_mod)
    monkeypatch.setitem(
        sys.modules, "langchain_community.tools.tavily_search", lc_comm_tavily_mod
    )

    # Fake langgraph.checkpoint.memory with MemorySaver
    langgraph_mod = types.ModuleType("langgraph")
    langgraph_checkpoint_mod = types.ModuleType("langgraph.checkpoint")
    langgraph_checkpoint_memory_mod = types.ModuleType("langgraph.checkpoint.memory")

    class FakeMemorySaver:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            container["memory_inits"].append({"args": args, "kwargs": kwargs})
            container["memory_instances"].append(self)

    langgraph_checkpoint_memory_mod.MemorySaver = FakeMemorySaver
    monkeypatch.setitem(sys.modules, "langgraph", langgraph_mod)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint", langgraph_checkpoint_mod)
    monkeypatch.setitem(
        sys.modules, "langgraph.checkpoint.memory", langgraph_checkpoint_memory_mod
    )

    # Fake langgraph.prebuilt with create_react_agent returning an executor with .stream
    langgraph_prebuilt_mod = types.ModuleType("langgraph.prebuilt")

    class PrintableMessage:
        def __init__(self):
            self.pretty_print_called = 0

        def pretty_print(self):
            self.pretty_print_called += 1
            container["pretty_print_calls"] += 1

    class FakeExecutor:
        def __init__(self):
            self.stream_invocations = []

        def stream(self, query, config, stream_mode=None):
            # record inputs for assertions
            container["stream_calls"].append(
                {"query": query, "config": config, "stream_mode": stream_mode}
            )
            # Yield two steps, each with a last message supporting pretty_print
            yield {"messages": ["ignored", PrintableMessage()]}
            yield {"messages": ["ignored-again", PrintableMessage()]}

    def fake_create_react_agent(model, tools, checkpointer=None, *args, **kwargs):
        container["create_agent_calls"].append(
            {"model": model, "tools": tools, "checkpointer": checkpointer, "args": args, "kwargs": kwargs}
        )
        return FakeExecutor()

    langgraph_prebuilt_mod.create_react_agent = fake_create_react_agent
    monkeypatch.setitem(sys.modules, "langgraph.prebuilt", langgraph_prebuilt_mod)

    return container


def import_fresh_lab_module(monkeypatch, tavily_value):
    # Ensure environment variable is set/unset as requested
    if tavily_value is None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TAVILY_API_KEY", tavily_value)

    # Build fake modules and capture container
    container = setup_fake_modules(monkeypatch)

    # Remove cached module if present to force top-level execution
    sys.modules.pop("langchain_agent_lab", None)

    # Import module under test; this runs its top-level code
    module = importlib.import_module("langchain_agent_lab")
    return module, container


@pytest.mark.parametrize("tavily_value", [None, "dummy-123"])
def test_agent_setup_and_streaming(monkeypatch, tavily_value):
    module, c = import_fresh_lab_module(monkeypatch, tavily_value)

    # 1) dotenv.load_dotenv was invoked once
    assert c["dotenv_called"] == 1

    # 2) ChatAnthropic initialized with expected model_name
    assert len(c["chatanthropic_inits"]) == 1
    chat_kwargs = c["chatanthropic_inits"][0]["kwargs"]
    assert chat_kwargs.get("model_name") == "claude-3-sonnet-20240229"
    assert len(c["model_instances"]) == 1
    model_instance = c["model_instances"][0]

    # 3) MemorySaver created exactly once and used as checkpointer
    assert len(c["memory_instances"]) == 1
    memory_instance = c["memory_instances"][0]

    # 4) TavilySearchResults initialized with max_results=2 and api_key from env (or None)
    assert len(c["search_instances"]) == 1
    tavily_call = c["tavily_inits"][0]
    t_kwargs = tavily_call["kwargs"]
    # Accept both positional/keyword; normalize
    max_results = t_kwargs.get("max_results", tavily_call["args"][0] if tavily_call["args"] else None)
    api_key = t_kwargs.get("api_key", tavily_call["args"][1] if len(tavily_call["args"]) > 1 else None)
    assert max_results == 2
    if tavily_value is None:
        assert api_key is None
    else:
        assert api_key == tavily_value
    search_instance = c["search_instances"][0]

    # 5) create_react_agent called with correct components
    assert len(c["create_agent_calls"]) == 1
    ca_call = c["create_agent_calls"][0]
    assert ca_call["model"] is model_instance
    assert ca_call["tools"] == [search_instance]
    assert ca_call["checkpointer"] is memory_instance

    # 6) Streaming invocation arguments
    assert len(c["stream_calls"]) == 1
    stream_call = c["stream_calls"][0]
    query = stream_call["query"]
    config = stream_call["config"]
    stream_mode = stream_call["stream_mode"]

    # Verify message payload contains a HumanMessage with expected content
    assert isinstance(query, dict)
    assert "messages" in query
    assert len(query["messages"]) == 1
    human_msg = query["messages"][0]
    # Should be instance of our fake HumanMessage class
    assert isinstance(human_msg, c["humanmessage_class"])
    assert human_msg.content == "hi im bob! and i live in sf"

    # Verify config thread_id and stream_mode
    assert config == {"configurable": {"thread_id": "abc123"}}
    assert stream_mode == "values"

    # 7) pretty_print invoked once per yielded step (our fake yields 2)
    assert c["pretty_print_calls"] == 2