import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema import BaseOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .llm_service import LLMService
from .knowledge_base_service import KnowledgeBaseService
from .web_search_service import WebSearchService
from .file_processor import FileProcessor
import asyncio
import time

load_dotenv()

class WorkflowExecutor:
    def __init__(self):
        self.llm_service = LLMService()
        self.kb_service = KnowledgeBaseService()
        self.web_search_service = WebSearchService()
        self.file_processor = FileProcessor()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    # ... keep existing code (validation methods)
    def validate_workflow(self, nodes: List[Any], edges: List[Dict]) -> Optional[str]:
        """Validate workflow configuration and return error message if invalid."""
        print(3)

        node_types = []
        print(nodes)

        for node in nodes:
            if isinstance(node, dict):
                component_type = node.get('data', {}).get('componentType')
            else:
                data = getattr(node, 'data', None)
                if isinstance(data, dict):
                    component_type = data.get('componentType')
                else:
                    component_type = getattr(data, 'componentType', None)
            node_types.append(component_type)

        print(node_types)

        if 'User Query' not in node_types:
            return "Workflow must contain a User Query component"

        if 'Output' not in node_types:
            return "Workflow must contain an Output component"

        print(4)

        for node in nodes:
            if isinstance(node, dict):
                node_data = node.get('data', {})
            else:
                node_data = getattr(node, 'data', None)
                if isinstance(node_data, dict):
                    pass
                else:
                    # For Pydantic NodeData
                    if hasattr(node_data, 'componentType') and hasattr(node_data, 'config'):
                        node_data = {
                            'componentType': node_data.componentType,
                            'config': node_data.config
                        }
                    else:
                        return "Invalid node data format"

            component_type = node_data.get('componentType')
            config = node_data.get('config', {})

            print(5)

            validation_error = self._validate_node_config(component_type, config)
            if validation_error:
                print(validation_error)
                print(6)
                return f"{component_type}: {validation_error}"

        print(7)

        if not self._has_valid_connections(nodes, edges):
            return "Workflow must have valid connections from User Query to Output"
        print(8)
        return None

    
    def _validate_node_config(self, component_type: str, config: Dict) -> Optional[str]:
        """Validate individual node configuration"""
        
        if component_type == 'Knowledge Base':
            if not config.get('apiKey'):
                return "API Key is required for Knowledge Base component"
            if not config.get('embeddingModel'):
                return "Embedding Model is required for Knowledge Base component"
        
        elif component_type == 'LLM (OpenAI)':
            if not config.get('apiKey'):
                return "API Key is required for LLM component"
            if not config.get('model'):
                return "Model selection is required for LLM component"
            if not config.get('prompt'):
                return "Prompt is required for LLM component"
        
        return None
    
    def _has_valid_connections(self, nodes: List[any], edges: List[Dict]) -> bool:
        """Check if workflow has valid connections"""

        user_query_node = None
        output_node = None
        print(1111)
        for node in nodes:
            component_type = getattr(node.data, 'componentType', None)
            if component_type == 'User Query':
                user_query_node = node.id
            elif component_type == 'Output':
                output_node = node.id

        if not user_query_node or not output_node:
            return False
        print(2222)
        # Check if there's a path from User Query to Output
        return self._has_path(user_query_node, output_node, edges)
    
    def _has_path(self, start: str, end: str, edges: List[any]) -> bool:
        """Check if there's a path from start node to end node"""

        # Build adjacency list
        graph = {}
        for edge in edges:
            source = edge.source
            target = edge.target
            if source not in graph:
                graph[source] = []
            graph[source].append(target)

        # DFS to find path
        visited = set()

        def dfs(node: str) -> bool:
            if node == end:
                return True
            if node in visited:
                return False

            visited.add(node)
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            return False

        return dfs(start)

    async def execute_workflow_langchain(self, workflow: Dict[str, Any], user_query: str) -> str:
        """Execute a workflow using LangChain chains and agents"""
        try:
            workflow_type = workflow.get('type', 'sequential')
            nodes = workflow.get('nodes', [])
            
            if workflow_type == 'sequential':
                return await self._execute_sequential_workflow(nodes, user_query)
            elif workflow_type == 'agent':
                return await self._execute_agent_workflow(nodes, user_query)
            elif workflow_type == 'parallel':
                return await self._execute_parallel_workflow(nodes, user_query)
            else:
                return await self._execute_custom_workflow(nodes, user_query)
                
        except Exception as e:
            return f"Error executing workflow: {str(e)}"

    async def _execute_sequential_workflow(self, nodes: List[Dict], user_query: str) -> str:
        """Execute nodes sequentially using LangChain SequentialChain"""
        try:
            chains = []
            input_variables = ["user_query"]
            output_variables = []
            
            for i, node in enumerate(nodes):
                node_type = node.get('type')
                config = node.get('config', {})
                
                if node_type == 'llm':
                    chain = await self._create_llm_chain(config, f"step_{i}")
                    chains.append(chain)
                    output_variables.append(f"step_{i}_output")
                elif node_type == 'knowledge_base':
                    chain = await self._create_kb_chain(config, f"step_{i}")
                    chains.append(chain)
                    output_variables.append(f"step_{i}_output")
                elif node_type == 'web_search':
                    chain = await self._create_web_search_chain(config, f"step_{i}")
                    chains.append(chain)
                    output_variables.append(f"step_{i}_output")
            
            if not chains:
                return "No valid chains created"
            
            # Create sequential chain
            sequential_chain = SequentialChain(
                chains=chains,
                input_variables=input_variables,
                output_variables=output_variables,
                verbose=True
            )
            
            # Execute chain
            result = await sequential_chain.ainvoke({"user_query": user_query})
            
            # Return the last output
            last_output_key = output_variables[-1] if output_variables else "output"
            return result.get(last_output_key, "No output generated")
            
        except Exception as e:
            return f"Error in sequential workflow: {str(e)}"

    async def _execute_agent_workflow(self, nodes: List[Dict], user_query: str) -> str:
        """Execute workflow using LangChain agents"""
        try:
            # Create tools from workflow nodes
            tools = []
            
            for node in nodes:
                tool = await self._create_tool_from_node(node)
                if tool:
                    tools.append(tool)
            
            if not tools:
                return "No tools created from workflow nodes"
            
            # Get LLM for agent
            llm = self.llm_service._get_model_instance('GPT 4o - Mini', 0.7)
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that can use various tools to answer questions."),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent
            agent = create_openai_functions_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            # Execute agent
            result = await agent_executor.ainvoke({"input": user_query})
            return result.get("output", "No output generated")
            
        except Exception as e:
            return f"Error in agent workflow: {str(e)}"

    async def _execute_parallel_workflow(self, nodes: List[Dict], user_query: str) -> str:
        """Execute workflow nodes in parallel"""
        try:
            tasks = []
            for node in nodes:
                task = self._execute_single_node_langchain(node, user_query)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    combined_results.append(f"Node {i} error: {str(result)}")
                else:
                    combined_results.append(f"Node {i}: {result}")
            
            return "\n\n".join(combined_results)
            
        except Exception as e:
            return f"Error in parallel workflow: {str(e)}"

    async def _execute_custom_workflow(self, nodes: List[Dict], user_query: str) -> str:
        """Execute custom workflow logic"""
        try:
            # Implement custom workflow logic based on node dependencies
            context = {"user_query": user_query}
            
            for node in nodes:
                result = await self._execute_single_node_langchain(node, user_query, context)
                context[f"node_{node.get('id', 'unknown')}_output"] = result
            
            # Return the final result or combined results
            return context.get("final_output", str(context))
            
        except Exception as e:
            return f"Error in custom workflow: {str(e)}"

    async def _execute_single_node_langchain(self, node: Dict, user_query: str, context: Dict = None) -> str:
        """Execute a single workflow node using LangChain"""
        try:
            node_type = node.get('type')
            config = node.get('config', {})
            
            if node_type == 'llm':
                return await self.llm_service.generate_response(user_query, "", config)
            elif node_type == 'knowledge_base':
                stack_id = config.get('stack_id', 'default')
                kb_context = await self.kb_service.retrieve_context(stack_id, user_query, config)
                return await self.llm_service.generate_response(user_query, kb_context, config)
            elif node_type == 'web_search':
                search_query = config.get('query', user_query)
                search_results = await self.web_search_service.search(search_query)
                return f"Web search results: {search_results}"
            elif node_type == 'file_processor':
                # Process files and return summary
                return "File processing completed"
            else:
                return f"Unknown node type: {node_type}"
                
        except Exception as e:
            return f"Error executing node: {str(e)}"

    async def _create_llm_chain(self, config: Dict, output_key: str) -> LLMChain:
        """Create LLM chain for workflow"""
        model_name = config.get('model', 'GPT 4o - Mini')
        llm = self.llm_service._get_model_instance(model_name, config.get('temperature', 0.7))
        
        prompt_template = config.get('prompt', 'Answer the following question: {user_query}')
        prompt = PromptTemplate(
            input_variables=["user_query"],
            template=prompt_template,
            output_key=f"{output_key}_output"
        )
        
        return LLMChain(llm=llm, prompt=prompt, output_key=f"{output_key}_output")

    async def _create_kb_chain(self, config: Dict, output_key: str) -> LLMChain:
        """Create knowledge base chain"""
        stack_id = config.get('stack_id', 'default')
        model_name = config.get('model', 'GPT 4o - Mini')
        llm = self.llm_service._get_model_instance(model_name, config.get('temperature', 0.7))
        
        # Create retrieval QA chain
        qa_chain = self.kb_service.create_retrieval_qa_chain(stack_id, llm, config)
        
        if qa_chain:
            return qa_chain
        else:
            # Fallback to simple LLM chain
            prompt = PromptTemplate(
                input_variables=["user_query"],
                template="Answer based on available knowledge: {user_query}",
                output_key=f"{output_key}_output"
            )
            return LLMChain(llm=llm, prompt=prompt, output_key=f"{output_key}_output")

    async def _create_web_search_chain(self, config: Dict, output_key: str) -> LLMChain:
        """Create web search chain"""
        model_name = config.get('model', 'GPT 4o - Mini')
        llm = self.llm_service._get_model_instance(model_name, config.get('temperature', 0.7))
        
        prompt = PromptTemplate(
            input_variables=["user_query"],
            template="Search the web for: {user_query} and provide a summary of findings.",
            output_key=f"{output_key}_output"
        )
        
        return LLMChain(llm=llm, prompt=prompt, output_key=f"{output_key}_output")

    async def _create_tool_from_node(self, node: Dict) -> Optional[Tool]:
        """Create LangChain tool from workflow node"""
        try:
            node_type = node.get('type')
            node_name = node.get('name', f"{node_type}_tool")
            config = node.get('config', {})
            
            if node_type == 'llm':
                async def llm_tool(query: str) -> str:
                    return await self.llm_service.generate_response(query, "", config)
                
                return Tool(
                    name=node_name,
                    description=f"Use this tool to generate responses using {config.get('model', 'LLM')}",
                    func=llm_tool
                )
            
            elif node_type == 'knowledge_base':
                async def kb_tool(query: str) -> str:
                    stack_id = config.get('stack_id', 'default')
                    context = await self.kb_service.retrieve_context(stack_id, query, config)
                    return await self.llm_service.generate_response(query, context, config)
                
                return Tool(
                    name=node_name,
                    description="Use this tool to search the knowledge base for relevant information",
                    func=kb_tool
                )
            
            elif node_type == 'web_search':
                async def web_search_tool(query: str) -> str:
                    return await self.web_search_service.search(query)
                
                return Tool(
                    name=node_name,
                    description="Use this tool to search the web for current information",
                    func=web_search_tool
                )
            
            return None
            
        except Exception as e:
            print(f"Error creating tool from node: {e}")
            return None

    async def create_conversation_chain(self, config: Dict[str, Any]):
        """Create a conversational chain with memory"""
        model_name = config.get('model', 'GPT 4o - Mini')
        llm = self.llm_service._get_model_instance(model_name, config.get('temperature', 0.7))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.get('prompt', 'You are a helpful assistant.')),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        return chain

    # ... keep existing code (original execution methods for backwards compatibility)
    async def execute_workflow(self, nodes: List[Dict], edges: List[Dict], user_query: str, stack_id: str = None) -> str:
        """
        Build execution order and run nodes.
        """
        print("[Executor] Building execution order...")
        execution_order = self._build_execution_order(nodes, edges)

        print(f"[Executor] Execution order: {execution_order}")

        context = {
            "user_query": user_query,
            "stack_id": stack_id
        }

        workflow_start = time.perf_counter()
        print(47)
        for node_id in execution_order:
            node = next((n for n in nodes if n.id == node_id), None)
            if not node:
                continue

            node_start = time.perf_counter()
            result = await self._execute_node(node, context)
            context[node_id] = result
            print(f"[Executor] Node {node_id} executed in {time.perf_counter() - node_start:.2f}s → {result[:50]}...")

        print(f"[Executor] Workflow executed in {time.perf_counter() - workflow_start:.2f}s")

        # Find final output
        output_result = context.get("user_query", user_query)
        for node_id in reversed(execution_order):
            if node_id in context and node_id not in ["user_query", "stack_id"]:
                output_result = context[node_id]
                break

        return output_result

        
    def _build_execution_order(self, nodes: List[any], edges: List[any]) -> List[str]:
        """Build topological order for workflow execution"""

        # Build adjacency list and in-degree count
        graph = {}
        in_degree = {}
        print(44)
        for node in nodes:
            node_id = node.id
            graph[node_id] = []
            in_degree[node_id] = 0
        print(45)
        for edge in edges:
            source = edge.source
            target = edge.target
            graph[source].append(target)
            in_degree[target] += 1

        # Topological sort
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        print(46)
        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        print(47)
        return result

    
    async def _execute_node(self, node: any, context: Dict) -> str:
        """
        Execute a single node safely.
        """
        component_type = getattr(node.data, 'componentType', None)
        config = getattr(node.data, 'config', {})
        node_id = getattr(node, 'id', '')
        print(node_id)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        try:
            if component_type == 'User Query':
                return context['user_query']

            elif component_type == 'Knowledge Base':
                user_query = context['user_query']
                print(f"[Node] Retrieving context for KB node {node_id}...")
                stack_id = context.get('stack_id', node_id.split('-')[0] if '-' in node_id else 'default')
                print(node_id)
                print("#####################################################")
                print(stack_id)
                print(user_query)
                

                print(f"[Node] Retrieving context for KB node {stack_id}...")
                return await self.kb_service.retrieve_context(node_id, user_query, config)

            elif component_type == 'LLM (OpenAI)':
                user_query = context['user_query']
                previous_context = ""

                for key, value in context.items():
                    if key not in ['user_query', 'stack_id', node_id] and isinstance(value, str):
                        previous_context += f"{value}\n"

                if config.get('webSearch', True) and config.get('serpApiKey'):
                    print(f"[Node] Performing web search for LLM node {node_id}...")
                    web_context = await self.web_search_service.get_search_context(user_query)
                    previous_context += f"\nWeb Search Results:\n{web_context}"

                print(f"[Node] Generating LLM response for node {node_id}...")
                return await self.llm_service.generate_response(
                    user_query,
                    previous_context.strip(),
                    config
                )

            elif component_type == 'Output':
                for key in reversed(list(context.keys())):
                    if key not in ['user_query', 'stack_id', node_id] and isinstance(context[key], str):
                        return context[key]
                return context.get('user_query', 'No output generated')

            return f"Unknown component type: {component_type}"

        except Exception as e:
            print(f"[Node] Error executing node {node_id}: {e}")
            return f"Error: {str(e)}"
        
    async def execute_workflow_chat(self, nodes: List[Dict], edges: List[Dict], user_query: str, stack_id: str = None) -> str:
        """
        Build execution order and run nodes.
        """
        print("[Executor] Building execution order...")
        execution_order = self._build_execution_order_chat(nodes, edges)

        print(f"[Executor] Execution order: {execution_order}")

        context = {
            "user_query": user_query,
            "stack_id": stack_id
        }

        workflow_start = time.perf_counter()
        print(47)
        for node_id in execution_order:
            # FIX: Use ["id"] instead of .id
            node = next((n for n in nodes if n["id"] == node_id), None)
            if not node:
                continue

            node_start = time.perf_counter()
            result = await self._execute_node_chat(node, context)
            context[node_id] = result
            print(f"[Executor] Node {node_id} executed in {time.perf_counter() - node_start:.2f}s → {result[:50]}...")

        print(f"[Executor] Workflow executed in {time.perf_counter() - workflow_start:.2f}s")

        # Find final output
        output_result = context.get("user_query", user_query)
        for node_id in reversed(execution_order):
            if node_id in context and node_id not in ["user_query", "stack_id"]:
                output_result = context[node_id]
                break
        print  (f"[Executor] Final output: {output_result}")
        return output_result
    
    def _build_execution_order_chat(self, nodes: List[dict], edges: List[dict]) -> List[str]:
        """Build topological order for workflow execution"""

        # Build adjacency list and in-degree count
        graph = {}
        in_degree = {}
        print(44)
        for node in nodes:
            node_id = node["id"]  # FIXED
            graph[node_id] = []
            in_degree[node_id] = 0

        print(45)
        for edge in edges:
            source = edge["source"]  # FIXED
            target = edge["target"]  # FIXED
            graph[source].append(target)
            in_degree[target] += 1

        # Topological sort (Kahn's algorithm)
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        print(46)
        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        print(47)
        return result

    async def _execute_node_chat(self, node: dict, context: Dict) -> str:
        """
        Execute a single node safely.
        """
        try:
            # Corrected attribute accesses
            node_data = node.get("data", {})
            component_type = node_data.get("componentType")
            config = node_data.get("config", {})
            node_id = node.get("id", "")

            print(node_id)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            if component_type == 'User Query':
                return context['user_query']

            elif component_type == 'Knowledge Base':
                user_query = context['user_query']
                stack_id = context.get('stack_id', node_id.split('-')[0] if '-' in node_id else 'default')
                print(node_id)
                print("#####################################################")
                print(stack_id)
                print(user_query)

                print(f"[Node] Retrieving context for KB node {stack_id}...")
                return await self.kb_service.retrieve_context(node_id, user_query, config)

            elif component_type == 'LLM (OpenAI)':
                user_query = context['user_query']
                previous_context = ""

                for key, value in context.items():
                    if key not in ['user_query', 'stack_id', node_id] and isinstance(value, str):
                        previous_context += f"{value}\n"

                if config.get('webSearch', True) and config.get('serpApiKey'):
                    print(f"[Node] Performing web search for LLM node {node_id}...")
                    web_context = await self.web_search_service.get_search_context(user_query)
                    previous_context += f"\nWeb Search Results:\n{web_context}"

                print(f"[Node] Generating LLM response for node {node_id}...")
                return await self.llm_service.generate_response(
                    user_query,
                    previous_context.strip(),
                    config
                )

            elif component_type == 'Output':
                for key in reversed(list(context.keys())):
                    if key not in ['user_query', 'stack_id', node_id] and isinstance(context[key], str):
                        return context[key]
                return context.get('user_query', 'No output generated')

            return f"Unknown component type: {component_type}"

        except Exception as e:
            print(f"[Node] Error executing node {node.get('id', '')}: {e}")
            return f"Error: {str(e)}"