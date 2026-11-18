import React, { useState, useEffect } from 'react';
import { Card } from '../components/Card';
import { EmptyState } from '../components/EmptyState';
import { Icon } from '../components/Icon';
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
      // All folders open by default
      const initialOpen = {};
      Object.keys(res.folders || {}).forEach((k) => {
        initialOpen[k] = true;
      });
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
      <>
        <header className="page-head">
          <div className="eyebrow">Exportable System Deliverables</div>
          <h1 className="page-title">
            Project <span className="gradient-ink">artifacts</span>
          </h1>
          <hr className="head-rule" />
        </header>
        <EmptyState
          icon="package"
          title="No artifacts yet"
          desc="Run an autonomous pipeline to generate sanitized datasets, feature sets, serialized models, a Jupyter notebook and executive reports — organized into 10 standardized folders."
        />
      </>
    );
  }

  return (
    <>
      <header className="page-head">
        <div className="eyebrow">Exportable System Deliverables</div>
        <h1 className="page-title">
          Project <span className="gradient-ink">artifacts</span>
        </h1>
        <p className="page-lede">
          Every stage of the autonomous pipeline produces auditable, standardized assets organized
          into 10 structured project directories.
        </p>
        <hr className="head-rule" />
      </header>

      {/* ZIP hero banner */}
      <section className="zip-banner">
        <div className="zip-banner-title">
          <Icon name="package" size={22} />
          Complete AI data science project package
        </div>
        <div className="zip-banner-desc">
          One self-contained archive with all <strong>10 standardized folders</strong>: raw &amp; cleaned CSV
          datasets, ML-ready matrices, the top serialized models (.pkl), an interactive Jupyter notebook
          (.ipynb), high-resolution visualizations (.png), the executive summary (.md) and full benchmark logs.
        </div>
        <div className="zip-banner-foot">
          <div className="zip-banner-id">Project identifier · {result.run_id}</div>
          <a href={getZipDownloadUrl(result.run_id)} className="btn btn-primary">
            <Icon name="download" size={16} />
            Download entire archive (.zip)
          </a>
        </div>
      </section>

      {/* Folder explorer */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <h3 className="section-title" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Icon name="folder" size={18} style={{ color: 'var(--ember-1)' }} />
          Browse &amp; download individual artifacts
        </h3>

        {loading ? (
          <div className="loading-block">
            <div className="spinner" />
            <span>Reading file system…</span>
          </div>
        ) : Object.keys(folders).length > 0 ? (
          Object.entries(folders).map(([folderName, folderData]) => {
            const isOpen = !!openFolders[folderName];
            const files = folderData.files || [];
            return (
              <div key={folderName} className="accordion folder-row">
                <button
                  type="button"
                  className="accordion-head"
                  onClick={() => toggleFolder(folderName)}
                  aria-expanded={isOpen}
                >
                  <span className="accordion-head-left">
                    <Icon name="folder" size={16} style={{ color: 'var(--ember-1)' }} />
                    <span className="mono" style={{ fontWeight: 600 }}>{folderName}/</span>
                    <span className="accordion-meta">— {folderData.description}</span>
                  </span>
                  <span className="accordion-meta">
                    {files.length} {files.length === 1 ? 'file' : 'files'}
                    <Icon
                      name="chevron-down"
                      size={15}
                      className={`accordion-chevron ${isOpen ? 'open' : ''}`}
                      style={{ marginLeft: 8, verticalAlign: -3 }}
                    />
                  </span>
                </button>

                {isOpen && files.length > 0 && (
                  <div className="folder-files">
                    {files.map((file, idx) => (
                      <div key={idx} className="file-row">
                        <div className="file-name">
                          <Icon name="file" size={13} style={{ flexShrink: 0, color: 'var(--ink-4)' }} />
                          <span>{file.path}</span>
                          <span className="file-size">({file.size_kb} KB)</span>
                        </div>
                        <a
                          href={getFileDownloadUrl(result.run_id, file.path)}
                          download={file.name}
                          className="btn btn-ghost btn-sm"
                          style={{ whiteSpace: 'nowrap' }}
                        >
                          <Icon name="download" size={13} />
                          Download
                        </a>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })
        ) : (
          <Card>
            <div style={{ textAlign: 'center', padding: 24, color: 'var(--ink-3)' }}>
              No individual artifacts were found for this run.
            </div>
          </Card>
        )}
      </section>
    </>
  );
}
