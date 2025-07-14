
import React from 'react';
import { MessageSquare, Database, Cpu, Monitor } from 'lucide-react';

const components = [
  {
    type: 'User Query',
    icon: MessageSquare,
    description: 'Enter point for querys'
  },
  {
    type: 'LLM (OpenAI)',
    icon: Cpu,
    description: 'Run a query with  LLM'
  },
  {
    type: 'Knowledge Base',
    icon: Database,
    description: 'Let LLM search info in your file'
  },
  {
    type: 'Output',
    icon: Monitor,
    description: 'Output of the result nodes as text'
  }
];

const ComponentPalette: React.FC = () => {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="w-64 bg-white border-r border-gray-200 p-4">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Components</h2>
      
      <div className="space-y-3">
        {components.map((component) => {
          const Icon = component.icon;
          return (
            <div
              key={component.type}
              className="flex items-center p-3 border border-gray-200 rounded-lg cursor-grab hover:bg-gray-50 transition-colors"
              draggable
              onDragStart={(event) => onDragStart(event, component.type)}
            >
              <Icon className="w-5 h-5 text-gray-600 mr-3" />
              <div>
                <div className="font-medium text-sm text-gray-900">{component.type}</div>
                <div className="text-xs text-gray-500">{component.description}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ComponentPalette;
