import React, { useState, useCallback } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { ArrowLeft, Play, MessageSquare, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import ComponentPalette from '@/components/ComponentPalette';
import ComponentConfigPanel from '@/components/ComponentConfigPanel';
import ChatInterface from '@/components/ChatInterface';
import CustomNode from '@/components/CustomNode';

const nodeTypes = {
  custom: CustomNode,
};

interface WorkflowBuilderProps {
  stackId: string;
  onBack: () => void;
}

const WorkflowBuilder: React.FC<WorkflowBuilderProps> = ({ stackId, onBack }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showChat, setShowChat] = useState(false);
  const { toast } = useToast();

  // Load existing workflow when component mounts
  React.useEffect(() => {
    loadWorkflow();
  }, [stackId]);

  const loadWorkflow = async () => {
    try {
      const response = await fetch(`http://localhost:8000/workflows/${stackId}`);
      if (response.ok) {
        const workflow = await response.json();
        if (workflow && workflow.nodes && workflow.edges) {
          setNodes(workflow.nodes);
          setEdges(workflow.edges);
        }
      }
    } catch (error) {
      console.error('Error loading workflow:', error);
    }
  };

  const onConnect = useCallback(
    (params: Edge | Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const reactFlowBounds = event.currentTarget.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');

      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = {
        x: event.clientX - reactFlowBounds.left - 100,
        y: event.clientY - reactFlowBounds.top - 50,
      };

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type: 'custom',
        position,
        data: { 
          label: type,
          componentType: type,
          config: getDefaultConfig(type)
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const getDefaultConfig = (type: string) => {
    switch (type) {
      case 'User Query':
        return { placeholder: 'Enter your query here' };
      case 'Knowledge Base':
        return { embeddingModel: 'text-embedding-3-small', apiKey: '' };
      case 'LLM (OpenAI)':
        return { model: 'GPT 4o - Mini', apiKey: '', prompt: '', temperature: 0.75 };
      case 'Output':
        return {};
      default:
        return {};
    }
  };

  const updateNodeConfig = (nodeId: string, config: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, config } }
          : node
      )
    );
  };

  const validateWorkflow = () => {
    for (const node of nodes) {
      if (node.data.componentType === 'Knowledge Base') {
        if (!node.data.config.apiKey) {
          toast({
            title: "Validation Error",
            description: "Please provide API Key for Knowledge Base component",
            variant: "destructive",
          });
          return false;
        }
      }
      if (node.data.componentType === 'LLM (OpenAI)') {
        if (!node.data.config.apiKey) {
          toast({
            title: "Validation Error",
            description: "Please provide API Key for LLM component",
            variant: "destructive",
          });
          return false;
        }
        if (!node.data.config.prompt) {
          toast({
            title: "Validation Error",
            description: "Please provide prompt for LLM component",
            variant: "destructive",
          });
          return false;
        }
      }
      if (node.data.componentType === 'User Query') {
        if (!node.data.config.placeholder) {
          toast({
            title: "Validation Error",
            description: "Please provide query text for User Query component",
            variant: "destructive",
          });
          return false;
        }
      }
    }
    return true;
  };

  const saveWorkflow = async () => {
    if (!validateWorkflow()) {
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/workflows/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stack_id: stackId,
          nodes,
          edges
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        toast({
          title: "Error",
          description: `Error saving workflow: ${error.detail}`,
          variant: "destructive",
        });
        return;
      }

      toast({
        title: "Success",
        description: "Workflow saved successfully!",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: `Error saving workflow: ${error.message}`,
        variant: "destructive",
      });
    }
  };

  const buildStack = async () => {
    if (!validateWorkflow()) {
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/workflows/build', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stack_id: stackId,
          nodes,
          edges
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        toast({
          title: "Error",
          description: `Error building stack: ${error.detail}`,
          variant: "destructive",
        });
        return;
      }

      const result = await response.json();
      
      // Update Output nodes with the result
      setNodes((nds) =>
        nds.map((node) =>
          node.data.componentType === 'Output'
            ? { 
                ...node, 
                data: { 
                  ...node.data, 
                  config: { 
                    ...node.data.config, 
                    output: result.result || 'Workflow executed successfully' 
                  } 
                } 
              }
            : node
        )
      );
      
      toast({
        title: "Success",
        description: "Stack built and executed successfully!",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: `Error building stack: ${error.message}`,
        variant: "destructive",
      });
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button variant="ghost" size="sm" onClick={onBack}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-sm">AI</span>
              </div>
              <h1 className="text-xl font-semibold text-gray-900">GenAI Stack</h1>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button variant="outline" onClick={buildStack}>
              <Play className="w-4 h-4 mr-2" />
              Build Stack
            </Button>
            <Button 
              onClick={() => setShowChat(true)}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <MessageSquare className="w-4 h-4 mr-2" />
              Chat with Stack
            </Button>
            <Button variant="outline" onClick={saveWorkflow}>Save</Button>
            <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center">
              <span className="text-white font-medium text-sm">S</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Component Palette */}
        <ComponentPalette />

        {/* Workflow Canvas */}
        <div className="flex-1 relative">
          {nodes.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-green-100 rounded-full flex items-center justify-center">
                  <Play className="w-8 h-8 text-green-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Drag & drop to get started</h3>
                <p className="text-gray-600">Build your AI workflow by dragging components from the left panel</p>
              </div>
            </div>
          ) : null}
          
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            fitView
          >
            <Controls />
            <MiniMap />
            <Background />
          </ReactFlow>
        </div>

        {/* Configuration Panel */}
        {selectedNode && (
          <ComponentConfigPanel
            node={selectedNode}
            onUpdateConfig={(config) => updateNodeConfig(selectedNode.id, config)}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>

      {/* Chat Interface */}
      {showChat && (
        <ChatInterface
          onClose={() => setShowChat(false)}
          workflow={{ nodes, edges, stackId }}
        />
      )}
      
    </div>
  );
};

export default WorkflowBuilder;