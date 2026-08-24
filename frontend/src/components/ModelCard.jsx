import React from 'react';
import { Card } from './Card';
import { getModelDownloadUrl } from '../api';

export function ModelCard({ model, runId }) {
  const { rank = 1, model_name = 'Model', metrics = {}, training_time = 0, selection_reason, filename, download_url } = model;
  const isRank1 = rank === 1;

  const actualFileName = filename || `model_${String(rank).padStart(2, '0')}_${(model_name || 'model').toLowerCase()}.pkl`;
  const actualDownloadUrl = download_url || (runId ? getModelDownloadUrl(runId, rank) : '#');

  return (
    <Card
      glow={isRank1 ? 'indigo' : ''}
      style={{
        padding: '20px 24px',
        border: isRank1 ? '1px solid rgba(99, 102, 241, 0.4)' : '1px solid var(--border-subtle)',
        background: isRank1 ? 'linear-gradient(180deg, rgba(99, 102, 241, 0.05) 0%, var(--surface-2) 100%)' : 'var(--surface-2)',
      }}
    >
      {/* Header: Rank Pill + Model Name + Latency */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', marginBottom: '8px' }}>
        <span
          style={{
            background: isRank1 ? '#2563eb' : 'var(--surface-4)',
            color: '#ffffff',
            padding: '3px 10px',
            borderRadius: '6px',
            fontSize: 'var(--text-xs)',
            fontWeight: 700,
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
          }}
        >
          RANK {rank}
        </span>
        <span style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          {model_name}
        </span>
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          (Training: {Number(training_time).toFixed(3)}s)
        </span>
      </div>

      {/* Selection Reason */}
      <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: '20px', lineHeight: 1.5 }}>
        <strong>Reason:</strong> {selection_reason || `Rank ${rank}: Top tier holdout cross-validation metric performance.`}
      </div>

      {/* Metrics Row + Download Button */}
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: '20px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '28px', alignItems: 'flex-start' }}>
          {Object.entries(metrics).map(([key, val]) => (
            <div key={key} style={{ minWidth: '80px' }}>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '2px' }}>
                {key}
              </div>
              <div style={{ fontSize: '1.45rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                {typeof val === 'number' ? val.toFixed(4) : (val || '—')}
              </div>
            </div>
          ))}
        </div>

        {runId && (
          <a
            href={actualDownloadUrl}
            download={actualFileName}
            className="btn btn-primary"
            style={{ fontSize: 'var(--text-xs)', padding: '8px 16px', borderRadius: 'var(--radius-sm)' }}
          >
            📥 Download ({actualFileName})
          </a>
        )}
      </div>
    </Card>
  );
}
