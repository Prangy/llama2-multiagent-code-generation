from typing import Callable, Dict, Any, Sequence, List, TypedDict, Annotated
import functools
import operator
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.llms import Ollama

import os
import subprocess

# Initialize the llama model
llm = Ollama(model="llama2")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "laanggraph practice"
os.environ["LANGCHAIN_API_KEY"] = "ls__8aff57aa62a54fea8768ad921350e898"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Base Agent class
class BaseAgent:
    def __init__(self, llm: Ollama, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt

    def create_executor(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.llm, [], prompt)  # Pass an empty list for tools
        executor = AgentExecutor(agent=agent, tools=[])
        return executor

    def create_node(self, name: str) -> Callable:
        raise NotImplementedError

# Specific Agent classes
class ProgrammerAgent(BaseAgent):
    def create_node(self, name: str) -> Callable:
        agent = self.create_executor()
        return functools.partial(agent_node, agent=agent, name=name)

class TesterAgent(BaseAgent):
    def create_node(self, name: str) -> Callable:
        agent = self.create_executor()
        return functools.partial(agent_node, agent=agent, name=name)

class DebuggerAgent(BaseAgent):
    def create_node(self, name: str) -> Callable:
        agent = self.create_executor()
        return functools.partial(agent_node, agent=agent, name=name)

class ExecutorAgent(BaseAgent):
    def create_node(self, name: str) -> Callable:
        agent = self.create_executor()
        return functools.partial(agent_node, agent=agent, name=name)

# Agent Factory
class AgentFactory:
    agent_types = {
        "Programmer": ProgrammerAgent,
        "Tester": TesterAgent,
        "Debugger": DebuggerAgent,
        "Executor": ExecutorAgent,
    }

    @staticmethod
    def get_agent(agent_type: str, llm: ChatOllama, system_prompt: str) -> BaseAgent:
        agent_class = AgentFactory.agent_types.get(agent_type)
        if not agent_class:
            raise ValueError(f"Agent type {agent_type} not recognized.")
        return agent_class(llm, system_prompt)

def agent_node(state, agent, name):
    
    while True:
        # Prompt user for input specific to each agent's role
        if name == "Programmer":
            user_input = input(">>> ")
        elif name == "Tester":
            user_input = input(">>> ")
        elif name == "Debugger":
            user_input = input(">>> ")
        elif name == "Executor":
            user_input = input(">>> ")

        if user_input.lower() == 'exit':
            break

        # Generate response using the language model
        output = llm.invoke(user_input)
        
        # Display Codellama response
        print("Agent's Output:")
        print(output)

        while True:
            # Ask the user if the output is satisfactory
            feedback = input("Was the agent's output satisfactory? (yes/no): ").lower()

            # Process user feedback
            if feedback == "yes":
                # If user is satisfied, proceed to the next agent
                state['messages'] = [HumanMessage(content=output, name=name)]
                next_agent = state.get("next", None)
                if next_agent:
                    result = agent.invoke(state)  # Invoke the agent to get the result
                    return result
                else:
                    return {"messages": [HumanMessage(content=output, name=name)]}
            elif feedback == "no":
                break  # Exit the inner loop
            else:
                print("Invalid feedback. Please enter 'yes' or 'no'.")
                # Continue the inner loop for another iteration

        # Continue the outer loop for another iteration

# Agent State and Workflow Setup
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Define the workflow creation function
def create_workflow(llm: Ollama):
    # Define system prompts for each agent
    programmer_system_prompt = '''**Role**: You are an expert Python programmer. Your task is to develop and generate efficient and readable Python code.
    **Task**: As a programmer, you need to complete the function. Utilize a systematic approach to break down the problem, create pseudocode, and then write well-commented code in Python.
    **Instructions**:
    1. **Understand and Clarify**: Ensure a thorough understanding of the task and requirements.
    2. **Algorithm/Method Selection**: Choose the most suitable algorithm or method for the task.
    3. **Pseudocode Creation**: Outline the steps in pseudocode before translating them into Python code.
    4. **Code Generation**: Write clean, efficient, and well-commented Python code.'''

    tester_system_prompt = '''**Role**: As a tester, your task is to create basic and comprehensive test cases based on the provided requirements and Python code. 
    **Task**: Your test cases should cover basic functionality as well as edge scenarios to ensure the code's robustness, reliability, and scalability.
    **Instructions**:
    - **Basic Test Cases**:
      - **Objective**: Verify basic functionality.
    - **Edge Test Cases**:
      - **Objective**: Evaluate the function's behavior under extreme or unusual conditions.
    - **Guidelines**:
      - Pay special attention to edge cases.
      - Keep test cases small and focused.
      - Avoid large-scale or medium-scale test cases.'''

    debugger_system_prompt = """**Role**: You are an expert in Python debugging. Your task is to analyze the given code and errors and generate code that handles them gracefully.
    **Instructions**:
    - Ensure the generated code is error-free.
    - Develop code capable of handling errors effectively."""

    executor_system_prompt = """**Role**: You are responsible for adding a testing layer to the Python code to execute it and validate its output.
    **Instructions**:
    - Verify that the code executes correctly and returns the expected output.
    - Implement testing procedures to catch any discrepancies between expected and actual outputs.
    - Provide feedback on any discrepancies or errors encountered during testing.
    **Python Code to Execute:**"""

    # Additional intelligent guidance can be added here for each agent's prompt
    
    programmer_agent = AgentFactory.get_agent("Programmer", llm, programmer_system_prompt)
    tester_agent = AgentFactory.get_agent("Tester", llm, tester_system_prompt)
    debugger_agent = AgentFactory.get_agent("Debugger", llm, debugger_system_prompt)
    executor_agent = AgentFactory.get_agent("Executor", llm, executor_system_prompt)

    programmer_node = programmer_agent.create_node("Programmer")
    tester_node = tester_agent.create_node("Tester")
    debugger_node = debugger_agent.create_node("Debugger")
    executor_node = executor_agent.create_node("Executor")

    # Define the agent state, edges, and graph
    workflow = StateGraph(AgentState)
    workflow.add_node("Programmer", programmer_node)
    workflow.add_node("Tester", tester_node)
    workflow.add_node("Debugger", debugger_node)
    workflow.add_node("Executor", executor_node)

    workflow.set_entry_point("Programmer")
    workflow.add_edge("Programmer", "Tester")
    workflow.add_edge("Debugger", "Executor")
    workflow.add_edge("Tester", "Executor")

    # Define the conditional function
    def decide_to_end(state):
        if 'errors' in state and state['errors']:
            return 'Debugger'
        else:
            return 'end'

    # Now, let's add conditional edges based on the flow of the process
    workflow.add_conditional_edges(
        "Executor",
        decide_to_end,
        {
            "end": END,
            "Debugger": "Debugger",
        },
    )

    # Compile and run the graph
    graph = workflow.compile()

    print("Welcome to Codellama Chat Interface")
    print("Type your question or command ('exit' to quit):")

    # Accept user input for human message content
    content = input("Enter the question you want to ask: ")

    # Run the graph
    for s in graph.stream({
        "messages": [HumanMessage(content=content)]
    }):
        if "__end__" not in s:
            print(s)
            print("----")

# Call the function to create the workflow
create_workflow(llm)
