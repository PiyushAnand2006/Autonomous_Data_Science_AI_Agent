import React from 'react';
import { Icon } from './Icon';
import { getModelDownloadUrl } from '../api';

export function ModelCard({ model, runId }) {
  const { rank = 1, model_name = 'Model', metrics = {}, training_time = 0, selection_reason, filename } = model;
  const isRank1 = rank === 1;

  const actualFileName = filename || `model_${String(rank).padStart(2, '0')}_${(model_name || 'model').toLowerCase()}.pkl`;
  const downloadUrl = runId ? getModelDownloadUrl(runId, rank) : '#';

  return (
    <div className={`card model-card ${isRank1 ? 'top1' : ''}`}>
      <div className="model-head">
        <span className={`rank-badge ${isRank1 ? 'gold' : ''}`}>Rank {rank}</span>
        <span className="model-name">{model_name}</span>
        <span className="model-time">trained in {Number(training_time).toFixed(3)}s</span>
        {isRank1 && (
          <span className="chip chip-ember" style={{ marginLeft: 'auto' }}>
            <Icon name="trophy" size={12} />
            Champion
          </span>
        )}
      </div>

      <div className="model-reason">
        <strong>Reason:</strong> {selection_reason || `Rank ${rank}: top-tier holdout cross-validation metric performance.`}
      </div>

      <div className="model-foot">
        <div className="model-metrics">
          {Object.entries(metrics).map(([key, val]) => (
            <div key={key} className="metric-cell">
              <div className="metric-key">{key}</div>
              <div className="metric-val">{typeof val === 'number' ? val.toFixed(4) : val || '—'}</div>
            </div>
          ))}
        </div>

        {runId && (
          <a href={downloadUrl} download={actualFileName} className="btn btn-secondary btn-sm">
            <Icon name="download" size={14} />
            {actualFileName}
          </a>
        )}
      </div>
    </div>
  );
}
