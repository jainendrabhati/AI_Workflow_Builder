import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import WorkflowBuilder from '@/components/WorkflowBuilder';

const WorkflowPage = () => {
  const { stackId } = useParams<{ stackId: string }>();
  const navigate = useNavigate();

  const handleBack = () => {
    navigate('/');
  };

  if (!stackId) {
    navigate('/');
    return null;
  }

  return (
    <WorkflowBuilder 
      stackId={stackId} 
      onBack={handleBack}
    />
  );
};

export default WorkflowPage;