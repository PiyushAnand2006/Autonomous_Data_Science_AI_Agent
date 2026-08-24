import React from 'react';

export function StatusPill({ mode = 'local' }) {
  const isAI = mode === 'ai';
  return (
    <span className={`status-pill ${isAI ? 'ai' : 'local'}`}>
      <span className="status-dot"></span>
      {isAI ? 'AI-POWERED' : 'LOCAL ENGINE'}
    </span>
  );
}
