import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import ChatInterface from './components/ChatInterface';
import DocumentUpload from './components/DocumentUpload';
import Sidebar from './components/Sidebar';
import { Menu, GraduationCap } from 'lucide-react';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true); // Default open on desktop
  const [currentWorkspace, setCurrentWorkspace] = useState('default'); // 'default' = Project Based Learning
  const [llmProvider, setLlmProvider] = useState('ollama');
  const [webSearch, setWebSearch] = useState(false);

  // Load documents when workspace changes
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/documents', {
          headers: { 'X-Workspace': currentWorkspace }
        });
        if (response.ok) {
          const data = await response.json();
          setDocuments(data.documents || []);
        }
      } catch (error) {
        console.error('Error fetching documents:', error);
      }
    };

    fetchDocuments();
    
    // Clear messages on workspace switch
    setMessages([]);
    addMessage('system', `Switched workspace to: ${currentWorkspace === 'default' ? 'Project Based Learning' : 'Teaching Methods'}.`);
  }, [currentWorkspace]);

  // Responsive sidebar handling
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };
    
    window.addEventListener('resize', handleResize);
    handleResize(); // Initial check
    
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const addMessage = (role, content) => {
    setMessages(prev => [...prev, { 
      id: Date.now(), 
      role, 
      content, 
      timestamp: new Date() 
    }]);
  };

  const handleSendMessage = async (message) => {
    if (!message.trim()) return;

    // Add user message
    addMessage('user', message);
    setIsLoading(true);

    try {
      // Format history for backend
      const history = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Workspace': currentWorkspace
        },
        body: JSON.stringify({ 
          query: message,
          llm_provider: llmProvider,
          web_search: webSearch,
          history: history
        }),
      });

      if (!response.ok) {
        throw new Error('Error sending request');
      }

      const data = await response.json();
      addMessage('assistant', data.response);
    } catch (error) {
      console.error('Error:', error);
      addMessage('assistant', 'Sorry, an error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDocumentUpload = async (files) => {
    // If it's a single file from previous logic, wrap in array (but new logic sends array)
    const fileList = Array.isArray(files) ? files : [files];
    
    // Add all files to list with 'uploading' status
    const newDocs = fileList.map(file => ({
      id: `temp_${file.name}_${Date.now()}`,
      name: file.name,
      status: 'uploading',
      uploadedAt: new Date(),
    }));

    setDocuments(prev => [...prev, ...newDocs]);
    setSidebarOpen(true); // Open sidebar to show progress
    
    addMessage('system', `🚀 Starting vectorization for ${fileList.length} document(s)...`);

    // Process files sequentially
    let successCount = 0;
    
    for (const file of fileList) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('http://localhost:8000/api/upload', {
          method: 'POST',
          headers: {
            'X-Workspace': currentWorkspace
          },
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Error uploading');
        }

        await response.json();
        
        // Update status to processed
        setDocuments(prev => prev.map(doc => 
          doc.name === file.name && doc.status === 'uploading'
            ? { ...doc, status: 'processed', id: Date.now() + Math.random() } 
            : doc
        ));
        
        successCount++;
      } catch (error) {
        console.error('Error:', error);
        // Update status to error
        setDocuments(prev => prev.map(doc => 
          doc.name === file.name && doc.status === 'uploading'
            ? { ...doc, status: 'error' }
            : doc
        ));
        addMessage('system', `⚠️ Error with "${file.name}": ${error.message}`);
      }
    }
    
    if (successCount > 0) {
      addMessage('system', `✅ ${successCount} of ${fileList.length} document(s) successfully vectorized and stored in knowledge base.`);
    }
    setIsLoading(false);
  };

  return (
    <div className="app">
      <Sidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)}
        documents={documents}
        currentWorkspace={currentWorkspace}
        onWorkspaceChange={setCurrentWorkspace}
        llmProvider={llmProvider}
        setLlmProvider={setLlmProvider}
        webSearch={webSearch}
        setWebSearch={setWebSearch}
      />
      <div className={`main-content ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <header className="app-header">
          <button 
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle Menu"
          >
            <Menu size={24} />
          </button>
          <div className="header-content">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <GraduationCap size={24} color="#4F46E5" />
              <h1>EduLearn RAG</h1>
            </div>
            <p className="subtitle">
              {currentWorkspace === 'default' ? 'Project Based Learning' : 'Teaching Methods'} Assistant
            </p>
          </div>
        </header>
        
        <div className="upload-section">
          <DocumentUpload onUpload={handleDocumentUpload} />
        </div>
        
        <ChatInterface 
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}

export default App;
