import React, { useState, useRef } from 'react';
import './Sidebar.css';
import { X, FileText, Database, Settings, BookOpen, ChevronDown } from 'lucide-react';

const Sidebar = ({ 
  isOpen, 
  onClose, 
  documents, 
  currentWorkspace, 
  onWorkspaceChange,
  llmProvider,
  setLlmProvider
}) => {
  return (
    <>
      {isOpen && <div className="sidebar-overlay" onClick={onClose} />}
      <div className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>Learning Projects</h2>
          <button className="close-button" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        
        <div className="sidebar-content">
          <section className="sidebar-section">
            <h3>
              <BookOpen size={14} />
              Active Project
            </h3>
            <div className="workspace-selector">
              <select 
                value={currentWorkspace} 
                onChange={(e) => onWorkspaceChange(e.target.value)}
                className="workspace-select"
              >
                <option value="default">Project Based Learning</option>
                <option value="teaching_methods">Teaching Methods</option>
              </select>
              <ChevronDown size={16} className="select-icon" />
            </div>
          </section>

          {/* Resources Section Hidden
          <section className="sidebar-section">
            <h3>
              <FileText size={14} />
              Resources ({documents.length})
            </h3>
            {documents.length === 0 ? (
              <p className="empty-state">No resources. Upload documents.</p>
            ) : (
              <ul className="document-list">
                {documents.map((doc) => (
                  <li key={doc.id} className="document-item">
                    <div className="document-icon">
                      <FileText size={16} />
                    </div>
                    <div className="document-details">
                      <span className="document-name">{doc.name}</span>
                      <span className="document-date">
                        {new Date(doc.uploadedAt).toLocaleDateString('en-US')}
                      </span>
                    </div>
                    <span className={`status-badge ${doc.status}`}></span>
                  </li>
                ))}
              </ul>
            )}
          </section>
          */}

          <section className="sidebar-section">
            <h3>
              <Settings size={14} />
              Configuration
            </h3>
            
            <div className="config-item">
              <label className="config-label">LLM Provider</label>
              <div className="select-wrapper">
                <select 
                  value={llmProvider}
                  onChange={(e) => setLlmProvider(e.target.value)}
                  className="config-select"
                >
                  <option value="huggingface">Kimi K2 Thinking (Reasoning)</option>
                  <option value="huggingface2">Meta Llama 3 8B Instruct</option>
                  <option value="ollama">Ollama (Local OSS 20B)</option>
                </select>
                <ChevronDown size={14} className="config-select-icon" />
              </div>
            </div>

            <div className="info-item">
              <span className="info-label">Vector Database</span>
              <span className="info-value">Supabase</span>
            </div>
          </section>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
