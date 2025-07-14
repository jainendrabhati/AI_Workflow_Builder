
import React from 'react';
import { ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface Stack {
  id: string;
  name: string;
  description: string;
  lastModified: Date;
}

interface StackCardProps {
  stack: Stack;
  onEdit: () => void;
}

const StackCard: React.FC<StackCardProps> = ({ stack, onEdit }) => {
  return (
    <Card className="p-6 hover:shadow-lg transition-shadow cursor-pointer bg-white">
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{stack.name}</h3>
          <p className="text-sm text-gray-600 mt-1">{stack.description}</p>
        </div>
        
        <div className="pt-4 border-t border-gray-100">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={onEdit}
            className="w-full justify-center"
          >
            Edit Stack
            <ExternalLink className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default StackCard;
