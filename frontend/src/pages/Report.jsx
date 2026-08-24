import React from 'react';
import { Card } from '../components/Card';

export function Report({ result }) {
  if (!result || !result.executive_summary) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📊</div>
        <div className="empty-state-title">No Executive Report Available</div>
        <div className="empty-state-desc">
          Run an autonomous pipeline to generate an in-depth analytical summary and Chief AI Scientist executive narrative.
        </div>
      </div>
    );
  }

  // Format inline markdown (bold, code, links)
  const formatInline = (str) => {
    if (!str) return str;
    const parts = [];
    let remaining = str;

    // Simple parser for **bold** and `code`
    let keyIdx = 0;
    while (remaining.length > 0) {
      const boldMatch = remaining.match(/\*\*(.*?)\*\*/);
      const codeMatch = remaining.match(/`(.*?)`/);

      let firstMatch = null;
      let matchType = null;

      if (boldMatch && (!codeMatch || boldMatch.index < codeMatch.index)) {
        firstMatch = boldMatch;
        matchType = 'bold';
      } else if (codeMatch) {
        firstMatch = codeMatch;
        matchType = 'code';
      }

      if (!firstMatch) {
        parts.push(remaining);
        break;
      }

      const matchIdx = firstMatch.index;
      if (matchIdx > 0) {
        parts.push(remaining.substring(0, matchIdx));
      }

      if (matchType === 'bold') {
        parts.push(<strong key={`b-${keyIdx++}`} style={{ color: 'var(--text-primary)' }}>{firstMatch[1]}</strong>);
      } else if (matchType === 'code') {
        parts.push(<code key={`c-${keyIdx++}`} style={{ background: 'var(--surface-3)', padding: '2px 6px', borderRadius: '4px', fontFamily: 'var(--font-mono)', fontSize: '0.85em', color: '#10b981' }}>{firstMatch[1]}</code>);
      }

      remaining = remaining.substring(matchIdx + firstMatch[0].length);
    }

    return parts;
  };

  // Full document parser handling tables, headers, lists, and paragraphs
  const renderFormattedMarkdown = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    const elements = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Markdown Table detection
      if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
        const tableLines = [];
        while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
          tableLines.push(lines[i].trim());
          i++;
        }

        if (tableLines.length >= 2) {
          const headerCols = tableLines[0].split('|').slice(1, -1).map((c) => c.trim());
          const isSeparator = tableLines[1].replace(/[-| :]/g, '').length === 0;
          const dataRows = isSeparator ? tableLines.slice(2) : tableLines.slice(1);

          elements.push(
            <div key={`table-${i}`} className="data-table-wrap" style={{ margin: '20px 0', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
              <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: 'var(--surface-3)', borderBottom: '1px solid var(--border-subtle)' }}>
                    {headerCols.map((h, hIdx) => (
                      <th key={hIdx} style={{ padding: '10px 14px', textAlign: 'left', fontWeight: 600, fontSize: 'var(--text-xs)', color: 'var(--text-primary)' }}>
                        {formatInline(h)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataRows.map((r, rIdx) => {
                    const cols = r.split('|').slice(1, -1).map((c) => c.trim());
                    return (
                      <tr key={rIdx} style={{ borderBottom: '1px solid var(--border-subtle)', background: rIdx % 2 === 0 ? 'var(--surface-1)' : 'transparent' }}>
                        {cols.map((c, cIdx) => (
                          <td key={cIdx} style={{ padding: '10px 14px', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                            {formatInline(c)}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
          continue;
        }
      }

      // Headers
      if (line.startsWith('# ')) {
        elements.push(<h1 key={i} style={{ fontSize: '1.75rem', fontWeight: 700, marginTop: '32px', marginBottom: '14px', color: 'var(--text-primary)' }}>{formatInline(line.slice(2))}</h1>);
      } else if (line.startsWith('## ')) {
        elements.push(<h2 key={i} style={{ fontSize: '1.4rem', fontWeight: 700, marginTop: '28px', marginBottom: '12px', color: 'var(--text-primary)' }}>{formatInline(line.slice(3))}</h2>);
      } else if (line.startsWith('### ')) {
        elements.push(<h3 key={i} style={{ fontSize: '1.15rem', fontWeight: 600, marginTop: '24px', marginBottom: '10px', color: 'var(--text-primary)' }}>{formatInline(line.slice(4))}</h3>);
      } else if (line.startsWith('#### ')) {
        elements.push(<h4 key={i} style={{ fontSize: '1.05rem', fontWeight: 600, marginTop: '20px', marginBottom: '8px', color: 'var(--text-primary)' }}>{formatInline(line.slice(5))}</h4>);
      } else if (line.startsWith('- ') || line.startsWith('* ')) {
        elements.push(
          <li key={i} style={{ marginLeft: '24px', marginBottom: '6px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {formatInline(line.slice(2))}
          </li>
        );
      } else if (/^\d+\.\s/.test(line)) {
        elements.push(
          <div key={i} style={{ marginLeft: '12px', marginBottom: '8px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {formatInline(line)}
          </div>
        );
      } else if (line.trim() === '---') {
        elements.push(<hr key={i} style={{ borderColor: 'var(--border-subtle)', margin: '28px 0' }} />);
      } else if (!line.trim()) {
        elements.push(<div key={i} style={{ height: '8px' }} />);
      } else {
        elements.push(
          <p key={i} style={{ marginBottom: '10px', color: 'var(--text-secondary)', lineHeight: 1.7, fontSize: '0.95rem' }}>
            {formatInline(line)}
          </p>
        );
      }

      i++;
    }

    return elements;
  };

  return (
    <div>
      <div style={{ marginBottom: '28px' }}>
        <div className="eyebrow">Strategic Executive Briefing</div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700, margin: '4px 0 8px 0' }}>
          🧠 AADS In-Depth Data Science & Machine Learning Analysis Report
        </h2>
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', fontSize: 'var(--text-xs)', color: 'var(--text-muted)', background: 'var(--surface-2)', padding: '12px 18px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', marginTop: '12px' }}>
          <div><strong>Run ID:</strong> <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{result.run_id}</span></div>
          <div><strong>Status:</strong> <span style={{ color: '#10b981', fontWeight: 600 }}>✓ Completed Successfully</span></div>
          <div><strong>Best Model:</strong> <span style={{ color: '#38bdf8', fontWeight: 600 }}>{result.best_model_name}</span></div>
        </div>
      </div>

      <Card style={{ padding: '32px' }}>
        <div className="markdown-body">
          {renderFormattedMarkdown(result.executive_summary)}
        </div>
      </Card>
    </div>
  );
}
