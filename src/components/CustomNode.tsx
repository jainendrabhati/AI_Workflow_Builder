
import React from 'react';
import { Handle, Position } from 'reactflow';
import { MessageSquare, Database, Cpu, Monitor } from 'lucide-react';

const getIcon = (type: string) => {
  switch (type) {
    case 'User Query':
      return MessageSquare;
    case 'Knowledge Base':
      return Database;
    case 'LLM (OpenAI)':
      return Cpu;
    case 'Output':
      return Monitor;
    default:
      return MessageSquare;
  }
};

const getColors = (type: string) => {
  switch (type) {
    case 'User Query':
      return 'bg-blue-500 border-blue-600';
    case 'Knowledge Base':
      return 'bg-purple-500 border-purple-600';
    case 'LLM (OpenAI)':
      return 'bg-green-500 border-green-600';
    case 'Output':
      return 'bg-orange-500 border-orange-600';
    default:
      return 'bg-gray-500 border-gray-600';
  }
};

interface CustomNodeProps {
  data: {
    label: string;
    componentType: string;
    config?: any;
  };
}

const CustomNode: React.FC<CustomNodeProps> = ({ data }) => {
  const Icon = getIcon(data.componentType);
  const colors = getColors(data.componentType);

  return (
    <div className={`px-4 py-2 border-2 rounded-lg bg-white shadow-md min-w-[150px] ${colors.replace('bg-', 'border-')}`}>
      <Handle type="target" position={Position.Top} className="w-3 h-3" />
      
      <div className="flex items-center space-x-2">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${colors}`}>
          <Icon className="w-4 h-4 text-white" />
        </div>
        <div>
          <div className="font-medium text-sm text-gray-900">{data.label}</div>
        </div>
      </div>
      
      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </div>
  );
};

export default CustomNode;
