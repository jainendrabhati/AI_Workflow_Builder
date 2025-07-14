
import React, { useEffect, useState } from 'react';
import { Plus } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import StackCard from '@/components/StackCard';
import CreateStackDialog from '@/components/CreateStackDialog';
import { useToast } from '@/hooks/use-toast';

interface Stack {
  id: string;
  name: string;
  description: string;
  lastModified: Date;
}

const Stacks = () => {
  const [stacks, setStacks] = useState<Stack[]>([]);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    fetchStacks();
  }, []);

  const fetchStacks = async () => {
    try {
      const response = await fetch('http://localhost:8000/stacks');
      if (response.ok) {
        const data = await response.json();
        setStacks(data);
      }
    } catch (error) {
      console.error('Error fetching stacks:', error);
    }
  };

  const handleCreateStack = async (name: string, description: string) => {
    try {
      const response = await fetch('http://localhost:8000/stacks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name,
          description,
        }),
      });

      if (response.ok) {
        const newStack = await response.json();
        setStacks([...stacks, newStack]);
        setShowCreateDialog(false);
        toast({
          title: "Success",
          description: "Stack created successfully!",
        });
      } else {
        toast({
          title: "Error",
          description: "Failed to create stack",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Error creating stack:', error);
      toast({
        title: "Error",
        description: "Failed to create stack",
        variant: "destructive",
      });
    }
  };

  const handleEditStack = (stackId: string) => {
    navigate(`/workflow/${stackId}`);
  };


  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-sm">AI</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-900">GenAI Stack</h1>
            </div>

            <div className="flex items-center space-x-4">
              <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center">
                <span className="text-white font-medium text-sm">S</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900">My Stacks</h2>
          <Button
            onClick={() => setShowCreateDialog(true)}
            className="bg-green-600 hover:bg-green-700 text-white"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Stack
          </Button>
        </div>

        {/* Stacks Grid */}
        {stacks.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 bg-green-100 rounded-full flex items-center justify-center">
              <Plus className="w-8 h-8 text-green-600" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Create New Stack</h3>
            <p className="text-gray-600 mb-6">
              Start building your generative AI apps with our essential tools and frameworks
            </p>
            <Button 
              onClick={() => setShowCreateDialog(true)}
              className="bg-green-600 hover:bg-green-700 text-white"
            >
              <Plus className="w-4 h-4 mr-2" />
              New Stack
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stacks.map((stack) => (
              <StackCard 
                key={stack.id} 
                stack={stack} 
                onEdit={() => handleEditStack(stack.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Create Stack Dialog */}
      <CreateStackDialog 
        open={showCreateDialog}
        onOpenChange={setShowCreateDialog}
        onCreateStack={handleCreateStack}
      />
    </div>
  );
};

export default Stacks;