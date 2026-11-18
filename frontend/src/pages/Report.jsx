import React from 'react';
import { Card } from '../components/Card';
import { EmptyState } from '../components/EmptyState';
import { Icon } from '../components/Icon';
import { getFileDownloadUrl } from '../api';

/* --- Inline markdown: bold, italic, code --- */
function formatInline(str) {
  if (!str) return str;
  const parts = [];
  let remaining = str;
  let keyIdx = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/\*\*(.*?)\*\*/);
    const codeMatch = remaining.match(/`(.*?)`/);
    const italicMatch = remaining.match(/(?<!\*)\*([^*]+)\*(?!\*)/);

    const matches = [];
    if (boldMatch) matches.push({ match: boldMatch, type: 'bold', index: boldMatch.index });
    if (codeMatch) matches.push({ match: codeMatch, type: 'code', index: codeMatch.index });
    if (italicMatch && (!boldMatch || italicMatch.index !== boldMatch.index)) {
      matches.push({ match: italicMatch, type: 'italic', index: italicMatch.index });
    }

    if (matches.length === 0) {
      parts.push(remaining);
      break;
    }

    matches.sort((a, b) => a.index - b.index);
    const first = matches[0];

    if (first.index > 0) {
      parts.push(remaining.substring(0, first.index));
    }

    if (first.type === 'bold') {
      parts.push(<strong key={`b-${keyIdx++}`}>{first.match[1]}</strong>);
    } else if (first.type === 'italic') {
      parts.push(<em key={`i-${keyIdx++}`}>{first.match[1]}</em>);
    } else if (first.type === 'code') {
      parts.push(<code key={`c-${keyIdx++}`}>{first.match[1]}</code>);
    }

    remaining = remaining.substring(first.index + first.match[0].length);
  }

  return parts;
}

/* --- Document parser: tables, headers, lists, rules, paragraphs --- */
function renderFormattedMarkdown(text) {
  if (!text) return null;
  const lines = text.split('\n');
  const elements = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Markdown tables
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
          <div key={`table-${i}`} className="table-wrap" style={{ margin: '18px 0' }}>
            <table className="data-table">
              <thead>
                <tr>
                  {headerCols.map((h, hIdx) => (
                    <th key={hIdx}>{formatInline(h)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataRows.map((r, rIdx) => {
                  const cols = r.split('|').slice(1, -1).map((c) => c.trim());
                  return (
                    <tr key={rIdx}>
                      {cols.map((c, cIdx) => (
                        <td key={cIdx}>{formatInline(c)}</td>
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

    if (line.startsWith('# ')) {
      elements.push(<h1 key={i}>{formatInline(line.slice(2))}</h1>);
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={i}>{formatInline(line.slice(3))}</h2>);
    } else if (line.startsWith('### ')) {
      elements.push(<h3 key={i}>{formatInline(line.slice(4))}</h3>);
    } else if (line.startsWith('#### ')) {
      elements.push(<h4 key={i}>{formatInline(line.slice(5))}</h4>);
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      elements.push(<li key={i}>{formatInline(line.slice(2))}</li>);
    } else if (/^\d+\.\s/.test(line)) {
      elements.push(
        <div key={i} style={{ margin: '0 0 8px 12px', color: 'var(--ink-2)', lineHeight: 1.65 }}>
          {formatInline(line)}
        </div>
      );
    } else if (line.trim() === '---') {
      elements.push(<hr key={i} />);
    } else if (!line.trim()) {
      elements.push(<div key={i} style={{ height: 8 }} />);
    } else {
      elements.push(<p key={i}>{formatInline(line)}</p>);
    }

    i++;
  }

  return elements;
}

export function Report({ result }) {
  if (!result || !result.executive_summary) {
    return (
      <>
        <header className="page-head">
          <div className="eyebrow">Strategic Executive Briefing</div>
          <h1 className="page-title">
            Executive <span className="gradient-ink">report</span>
          </h1>
          <hr className="head-rule" />
        </header>
        <EmptyState
          icon="report"
          title="No executive report yet"
          desc="Run an autonomous pipeline to generate the Chief AI Scientist's in-depth analytical narrative and strategic recommendations."
        />
      </>
    );
  }

  return (
    <>
      <header className="page-head">
        <div className="eyebrow">Strategic Executive Briefing</div>
        <h1 className="page-title">
          Executive <span className="gradient-ink">report</span>
        </h1>
        <p className="page-lede">
          In-depth data science &amp; machine-learning analysis authored by the Chief AI Report agent.
        </p>
        <hr className="head-rule" />
      </header>

      {/* Meta + downloads */}
      <div className="report-meta">
        <div className="report-meta-facts">
          <span>
            <strong>Run:</strong> <span className="mono">{result.run_id}</span>
          </span>
          <span>
            <strong>Status:</strong> <span style={{ color: 'var(--jade)', fontWeight: 600 }}>Completed</span>
          </span>
          <span>
            <strong>Best model:</strong> <span className="mono" style={{ color: 'var(--ember-3)', fontWeight: 600 }}>{result.best_model_name}</span>
          </span>
        </div>
        <div className="report-actions">
          <a
            href={getFileDownloadUrl(result.run_id, '08_Reports/executive_summary.pdf', true)}
            download="executive_summary.pdf"
            className="btn btn-secondary btn-sm"
          >
            <Icon name="download" size={14} />
            PDF
          </a>
          <a
            href={getFileDownloadUrl(result.run_id, '08_Reports/executive_summary.docx', true)}
            download="executive_summary.docx"
            className="btn btn-secondary btn-sm"
          >
            <Icon name="download" size={14} />
            DOCX
          </a>
          <a
            href={getFileDownloadUrl(result.run_id, '08_Reports/executive_summary.md', true)}
            download="executive_summary.md"
            className="btn btn-ghost btn-sm"
          >
            <Icon name="download" size={14} />
            Markdown
          </a>
        </div>
      </div>

      {/* Report body */}
      <Card style={{ padding: 34 }}>
        <div className="markdown-body">{renderFormattedMarkdown(result.executive_summary)}</div>
      </Card>
    </>
  );
}
