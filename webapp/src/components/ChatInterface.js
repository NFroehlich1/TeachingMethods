import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './ChatInterface.css';
import { Send, Bot, User, Sparkles } from 'lucide-react';

const ChatInterface = ({ messages, onSendMessage, isLoading }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <div className="welcome-icon">
              <Sparkles size={32} />
            </div>
            <h2>Welcome to EduLearn!</h2>
            <p>Your personal learning assistant. Upload course materials and ask questions to deepen your knowledge.</p>
            <div className="example-questions">
              <p className="example-title" style={{textAlign: 'left', fontSize: '0.8rem', fontWeight: '600', color: '#9ca3af', marginBottom: '0.5rem'}}>SUGGESTED QUESTIONS:</p>
              <button 
                className="example-btn"
                onClick={() => onSendMessage('Summarize the core concepts of this document.')}
              >
                💡 Summarize core concepts
              </button>
              <button 
                className="example-btn"
                onClick={() => onSendMessage('Create a quiz based on the content.')}
              >
                ❓ Create a content quiz
              </button>
              <button 
                className="example-btn"
                onClick={() => onSendMessage('Explain the difficult terms simply.')}
              >
                🎓 Explain difficult terms simply
              </button>
            </div>
          </div>
        )}
        
        {messages.map((message) => (
          <div 
            key={message.id} 
            className={`message ${message.role}`}
          >
            {message.role !== 'system' && (
              <div className="message-avatar">
                {message.role === 'user' ? <User size={20} /> : <Bot size={20} />}
              </div>
            )}
            
            <div className="message-content">
              <div className="message-bubble">
                {message.role === 'assistant' ? (
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table: ({ node, ...props }) => (
                        <div className="markdown-table-wrapper">
                          <table {...props} />
                        </div>
                      ),
                      th: ({ node, ...props }) => (
                        <th {...props} style={{ 
                          border: '1px solid #e5e7eb',
                          padding: '8px 12px',
                          backgroundColor: '#f9fafb',
                          fontWeight: '600',
                          textAlign: 'left'
                        }} />
                      ),
                      td: ({ node, ...props }) => (
                        <td {...props} style={{ 
                          border: '1px solid #e5e7eb',
                          padding: '8px 12px'
                        }} />
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  message.content
                )}
              </div>
              {message.role !== 'system' && (
                <div className="message-time">{formatTime(message.timestamp)}</div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant">
            <div className="message-avatar">
              <Bot size={20} />
            </div>
            <div className="message-content">
              <div className="message-bubble">
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your project..."
          className="chat-input"
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className="send-button"
          disabled={isLoading || !input.trim()}
        >
          <Send size={20} />
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
