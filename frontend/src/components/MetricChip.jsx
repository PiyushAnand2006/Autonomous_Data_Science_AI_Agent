import React from 'react';

export function MetricChip({ label, value }) {
  return (
    <span className="metric-chip">
      <span className="metric-chip-label">{label}</span>
      <span className="metric-chip-value">{value}</span>
    </span>
  );
}
