import React, { useState, useRef } from 'react';
import './DocumentUpload.css';
import { Upload, FileText, X, Plus, Play, Loader } from 'lucide-react';

const DocumentUpload = ({ onUpload }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFiles(files);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      handleFiles(files);
    }
  };

  const handleFiles = (files) => {
    const allowedTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'text/markdown',
    ];
    
    const validFiles = files.filter(file => 
      allowedTypes.includes(file.type) || file.name.match(/\.(pdf|doc|docx|txt|md)$/i)
    );

    if (validFiles.length !== files.length) {
      alert('Some files were skipped (unsupported type).');
    }

    // Add new files to existing selection, avoiding duplicates by name
    setSelectedFiles(prev => {
      const newFiles = validFiles.filter(newFile => 
        !prev.some(existing => existing.name === newFile.name)
      );
      return [...prev, ...newFiles];
    });
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRemove = (fileName) => {
    setSelectedFiles(prev => prev.filter(f => f.name !== fileName));
  };

  const handleUploadClick = async () => {
    if (selectedFiles.length === 0) return;
    
    setIsUploading(true);
    // Pass the files to parent for processing
    await onUpload(selectedFiles);
    // Clear selection after upload starts (or parent handles cleanup)
    setSelectedFiles([]);
    setIsUploading(false);
  };

  return (
    <div className="document-upload-container">
      {selectedFiles.length === 0 ? (
        <div
          className={`upload-area ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-icon">
            <Plus size={24} />
          </div>
          <div className="upload-content">
            <h3>Add Material</h3>
            <p className="upload-hint">
              PDF, DOCX, TXT, MD (Drag & Drop, multiple files)
            </p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileSelect}
            accept=".pdf,.doc,.docx,.txt,.md"
            multiple
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div className="upload-queue">
          <div className="queue-header">
            <h3>{selectedFiles.length} file(s) selected</h3>
            <div className="queue-actions">
              <button 
                className="add-more-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                <Plus size={16} />
              </button>
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileSelect}
                accept=".pdf,.doc,.docx,.txt,.md"
                multiple
                style={{ display: 'none' }}
              />
            </div>
          </div>
          
          <div className="file-list">
            {selectedFiles.map((file) => (
              <div key={file.name} className="queue-item">
                <FileText size={16} className="file-icon" />
                <span className="file-name">{file.name}</span>
                <span className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                <button 
                  className="remove-file-btn"
                  onClick={() => handleRemove(file.name)}
                  disabled={isUploading}
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>

          <button 
            className={`upload-all-btn ${isUploading ? 'loading' : ''}`}
            onClick={handleUploadClick}
            disabled={isUploading}
          >
            {isUploading ? (
              <>
                <Loader size={18} className="spin" />
                Processing...
              </>
            ) : (
              <>
                <Play size={18} />
                Start Vectorization
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};

export default DocumentUpload;
