import React, { useState, useEffect } from 'react';
import { X, Eye, EyeOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { useToast } from '@/hooks/use-toast';

interface ComponentConfigPanelProps {
  node: any;
  onUpdateConfig: (config: any) => void;
  onClose: () => void;
}

const ComponentConfigPanel: React.FC<ComponentConfigPanelProps> = ({
  node,
  onUpdateConfig,
  onClose
}) => {
  const [config, setConfig] = useState(node.data.config || {});
  const [showApiKey, setShowApiKey] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string>(node.data.config?.fileName || '');
  const { toast } = useToast();

  useEffect(() => {
    setConfig(node.data.config || {});
  }, [node]);

  useEffect(() => {
    console.log("[ComponentConfigPanel] componentType:", node?.data?.componentType);
}, [node]);

  const updateConfig = (key: string, value: any) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    onUpdateConfig(newConfig);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      toast({
        title: "Invalid File",
        description: "Please upload a PDF file only",
        variant: "destructive",
      });
      return;
    }

    if (!config.apiKey) {
      toast({
        title: "API Key Required",
        description: "Please provide API Key before uploading file",
        variant: "destructive",
      });
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('api_key', config.apiKey);
      formData.append('embedding_model', config.embeddingModel || 'text-embedding-3-small');

      const response = await fetch(`http://localhost:8000/upload-file/${node.id}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        toast({
          title: "Upload Failed",
          description: `Upload failed: ${error.detail}`,
          variant: "destructive",
        });
        return;
      }

      const result = await response.json();
      setUploadedFileName(file.name);
      updateConfig('fileName', file.name);
      updateConfig('fileUploaded', true);
      toast({
        title: "Upload Success",
        description: `File uploaded successfully! Processed ${result.chunks_count} chunks.`,
      });
    } catch (error) {
      toast({
        title: "Upload Failed",
        description: `Upload failed: ${error.message}`,
        variant: "destructive",
      });
    }
  };

  const renderConfigFields = () => {
    switch (node.data.componentType) {
      case 'User Query':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Enter point for querys
              </label>
              <Input
                placeholder="User Query"
                value={config.placeholder || ''}
                onChange={(e) => updateConfig('placeholder', e.target.value)}
              />
            </div>
          </div>
        );

      case 'Knowledge Base':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Let LLM search info in your file
              </label>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                File for Knowledge Base
              </label>
              {uploadedFileName && (
                <div className="mb-2 text-sm text-green-600 bg-green-50 p-2 rounded border">
                  📄 Uploaded: {uploadedFileName}
                </div>
              )}
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center">
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Button variant="outline" size="sm" type="button" asChild>
                    <span>Upload PDF File</span>
                  </Button>
                </label>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Embedding Model
              </label>
              <Select
                value={config.embeddingModel || 'text-embedding-3-large'}
                onValueChange={(value) => updateConfig('embeddingModel', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="text-embedding-3-large">text-embedding-3-large</SelectItem>
                  <SelectItem value="text-embedding-3-small">text-embedding-3-small</SelectItem>
                  <SelectItem value="gemini-1.5-flash">gemini-1.5-flash</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                API Key
              </label>
              <div className="relative">
                <Input
                  type={showApiKey ? 'text' : 'password'}
                  value={config.apiKey || ''}
                  onChange={(e) => updateConfig('apiKey', e.target.value)}
                  placeholder="••••••••••••••••"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute right-2 top-1/2 transform -translate-y-1/2"
                  onClick={() => setShowApiKey(!showApiKey)}
                >
                  {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </Button>
              </div>
            </div>
          </div>
        );

      case 'LLM (OpenAI)':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Run a query with OpenAI LLM
              </label>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model
              </label>
              <Select
                value={config.model || 'GPT 4o - Mini'}
                onValueChange={(value) => updateConfig('model', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="GPT 4o - Mini">"GPT 4o - Mini"</SelectItem>
                  <SelectItem value="GPT 4o">GPT 4o</SelectItem>
                  <SelectItem value="GPT 3.5 Turbo">GPT 3.5 Turbo</SelectItem>
                  <SelectItem value="gemini-1.5-flash">gemini-1.5-flash </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                API Key
              </label>
              <div className="relative">
                <Input
                  type={showApiKey ? 'text' : 'password'}
                  value={config.apiKey || ''}
                  onChange={(e) => updateConfig('apiKey', e.target.value)}
                  placeholder="••••••••••••••••"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute right-2 top-1/2 transform -translate-y-1/2"
                  onClick={() => setShowApiKey(!showApiKey)}
                >
                  {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </Button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Prompt
              </label>
              <Textarea
                value={config.prompt}
                onChange={(e) => updateConfig('prompt', e.target.value)}
                rows={3}
                placeholder="Enter your prompt"
              />
              <div className="text-xs text-blue-600 mt-1">
                <span className="font-medium">CONTEXT</span> (content) <span className="font-medium">User Query</span> (query)
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Temperature: {config.temperature || 0.75}
              </label>
              <Slider
                value={[config.temperature || 0.75]}
                onValueChange={(value) => updateConfig('temperature', value[0])}
                max={1}
                min={0}
                step={0.01}
                className="mt-2"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">
                WebSearch Tool
              </label>
              <Switch
                checked={config.webSearch || true}
                onCheckedChange={(checked) => updateConfig('webSearch', checked)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                SERP API
              </label>
              <div className="relative">
                <Input
                  type={showApiKey ? 'text' : 'password'}
                  value={config.serpApiKey || ''}
                  onChange={(e) => updateConfig('serpApiKey', e.target.value)}
                  placeholder="••••••••••••••••"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute right-2 top-1/2 transform -translate-y-1/2"
                  onClick={() => setShowApiKey(!showApiKey)}
                >
                  {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </Button>
              </div>
            </div>
          </div>
        );

      case 'Output':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Output of the result nodes as text
              </label>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Output Text
              </label>
               <div className="text-sm text-gray-800 p-3 bg-gray-50 rounded border min-h-[100px] max-h-[300px] overflow-y-auto whitespace-pre-wrap">
                {config.output || "No output yet"}
              </div>
            </div>
          </div>
        );

      default:
        return <div>No configuration available for this component.</div>;
    }
  };

  return (
    <div className="w-80 bg-white border-l border-gray-200 p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{node.data.componentType}</h3>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="w-4 h-4" />
        </Button>
      </div>
      
      {renderConfigFields()}
    </div>
  );
};

export default ComponentConfigPanel;