import React, { useState, useEffect } from 'react';
import { Card } from '../components/Card';
import { listPipelineFiles, getFileDownloadUrl, getZipDownloadUrl } from '../api';

export function Artifacts({ result }) {
  const [folders, setFolders] = useState({});
  const [loading, setLoading] = useState(false);
  const [openFolders, setOpenFolders] = useState({});

  useEffect(() => {
    if (result && result.run_id) {
      loadFiles(result.run_id);
    } else {
      setFolders({});
    }
  }, [result]);

  const loadFiles = async (runId) => {
    setLoading(true);
    try {
      const res = await listPipelineFiles(runId);
      setFolders(res.folders || {});
      // Open all by default
      const initialOpen = {};
      Object.keys(res.folders || {}).forEach((k) => { initialOpen[k] = true; });
      setOpenFolders(initialOpen);
    } catch (e) {
      console.error('Failed to load files', e);
    } finally {
      setLoading(false);
    }
  };

  const toggleFolder = (name) => {
    setOpenFolders((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  if (!result) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📦</div>
        <div className="empty-state-title">No Artifacts Available</div>
        <div className="empty-state-desc">
          Run an autonomous pipeline to generate sanitized datasets, feature sets, serialized models, Jupyter notebooks, and reports.
        </div>
      </div>
    );
  }

  return (
    <div>
      <div style={{ marginBottom: '32px' }}>
        <div className="eyebrow">Exportable System Deliverables</div>
        <h2>Project Artifacts Explorer</h2>
        <p>
          Every stage of the autonomous pipeline produces auditable, standardized assets organized into 10 structured project directories.
        </p>
      </div>

      {/* Hero ZIP Download Banner */}
      <div className="hero-banner" style={{ marginBottom: '32px' }}>
        <div className="hero-banner-title">📦 Complete AI Data Science Project Package</div>
        <div className="hero-banner-desc">
          Download the complete self-contained project archive containing all <strong>10 standardized folders</strong>:
          raw & cleaned CSV datasets, ML matrices, 4 top serialized models (.pkl), interactive Jupyter notebook (.ipynb), high-res visualizations (.png), executive summary (.md), and full benchmark logs.
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', marginTop: '16px' }}>
          <div className="hero-banner-id">Project Identifier: {result.run_id}</div>
          <a
            href={getZipDownloadUrl(result.run_id)}
            className="btn btn-primary"
          >
            ⬇️ Download Entire Project Archive (.ZIP)
          </a>
        </div>
      </div>

      {/* Folder by Folder Explorer */}
      <h3 style={{ marginBottom: '16px' }}>📂 Browse & Download Individual Artifacts</h3>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
          Reading file system...
        </div>
      ) : Object.keys(folders).length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {Object.entries(folders).map(([folderName, folderData]) => {
            const isOpen = !!openFolders[folderName];
            return (
              <Card key={folderName}>
                <div
                  onClick={() => toggleFolder(folderName)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    userSelect: 'none',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '1.1rem' }}>{isOpen ? '📂' : '📁'}</span>
                    <strong>{folderName}/</strong>
                    <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                      ({folderData.files?.length || 0} files) — <em>{folderData.description}</em>
                    </span>
                  </div>
                  <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>
                    {isOpen ? '▲' : '▼'}
                  </span>
                </div>

                {isOpen && folderData.files && folderData.files.length > 0 && (
                  <div style={{ marginTop: '16px', borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                    {folderData.files.map((file, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '6px 0',
                          borderBottom: idx < folderData.files.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: 'var(--text-muted)' }}>📄</span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-primary)' }}>
                            {file.path}
                          </span>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                            ({file.size_kb} KB)
                          </span>
                        </div>

                        <a
                          href={getFileDownloadUrl(result.run_id, file.path)}
                          download={file.name}
                          className="btn btn-ghost"
                          style={{ padding: '4px 12px', fontSize: 'var(--text-xs)', whiteSpace: 'nowrap' }}
                        >
                          📥 Download {file.name}
                        </a>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      ) : (
        <Card>
          <div style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
            No individual artifacts found for this run.
          </div>
        </Card>
      )}
    </div>
  );
}
