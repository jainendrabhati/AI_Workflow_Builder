from typing import List, Dict, Any, Optional
import openai
import os
from dotenv import load_dotenv
from .llm_service import LLMService
from .knowledge_base_service import KnowledgeBaseService
from .web_search_service import WebSearchService
import asyncio
import time
from typing import List, Dict

load_dotenv()

class WorkflowExecutor:
    def __init__(self):
        self.llm_service = LLMService()
        self.kb_service = KnowledgeBaseService()
        self.web_search_service = WebSearchService()
    

    
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
