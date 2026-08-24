import React, { useEffect, useRef } from 'react';
import { Card } from './Card';

export function PipelineProgress({ logs, status, isRunning, runId, error }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const isError = status === 'error' || Boolean(error);

  return (
    <Card glow={isRunning ? 'sapphire' : isError ? '' : status === 'completed' ? 'emerald' : ''} style={{ marginTop: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            className={`phase-dot ${isRunning ? 'active' : isError ? '' : status === 'completed' ? 'done' : ''}`}
            style={isError ? { background: 'var(--accent-rose)', boxShadow: '0 0 10px rgba(244, 63, 94, 0.5)' } : {}}
          />
          <h4 style={{ margin: 0, color: isError ? 'var(--accent-rose)' : 'var(--text-primary)' }}>
            {isRunning
              ? 'Autonomous Multi-Agent Pipeline Running...'
              : isError
              ? 'Pipeline Error'
              : status === 'completed'
              ? 'Pipeline Completed Successfully'
              : 'Pipeline Execution'}
          </h4>
        </div>
        {runId && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            RUN ID: {runId}
          </span>
        )}
      </div>

      <div
        className="pipeline-progress"
        style={{
          maxHeight: '260px',
          overflowY: 'auto',
          background: 'var(--surface-1)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        {logs.map((log, idx) => {
          const isLatest = idx === logs.length - 1;
          return (
            <div key={idx} className="pipeline-phase">
              <div
                className={`phase-dot ${isLatest && isRunning ? 'active' : isLatest && isError ? '' : 'done'}`}
                style={isLatest && isError ? { background: 'var(--accent-rose)' } : {}}
              />
              <div
                className={`phase-text ${isLatest && isRunning ? 'active' : ''}`}
                style={{
                  fontFamily: 'var(--font-mono)',
                  color: isLatest && isError ? 'var(--accent-rose)' : undefined,
                }}
              >
                {log}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      {isError && error && (
        <div
          style={{
            marginTop: '12px',
            padding: '12px 16px',
            background: 'rgba(244, 63, 94, 0.08)',
            border: '1px solid rgba(244, 63, 94, 0.25)',
            borderRadius: 'var(--radius-md)',
            color: '#fca5a5',
            fontSize: 'var(--text-sm)',
            fontFamily: 'var(--font-mono)',
          }}
        >
          <strong>Error Details:</strong> {error}
        </div>
      )}
    </Card>
  );
}
