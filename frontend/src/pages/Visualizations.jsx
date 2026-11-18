import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Card } from '../components/Card';
import { EmptyState } from '../components/EmptyState';
import { Icon } from '../components/Icon';
import { listVisualizations, getFileDownloadUrl, API_BASE } from '../api';

export function Visualizations({ result }) {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeCategory, setActiveCategory] = useState('ALL');
  const [selectedImg, setSelectedImg] = useState(null);

  useEffect(() => {
    if (result && result.run_id) {
      loadVisualizations(result.run_id);
    } else {
      setImages([]);
      setSelectedImg(null);
    }
  }, [result]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setSelectedImg(null);
    };
    if (selectedImg) window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedImg]);

  const loadVisualizations = async (runId) => {
    setLoading(true);
    try {
      const res = await listVisualizations(runId);
      setImages(res.images || []);
    } catch (e) {
      console.error('Failed to load visualizations', e);
    } finally {
      setLoading(false);
    }
  };

  const getFullImgUrl = (img, isDownload = false) => {
    if (!img) return '';
    if (img.url && img.url.startsWith('http')) {
      const sep = img.url.includes('?') ? '&' : '?';
      return isDownload ? `${img.url}${sep}download=true` : `${img.url}${sep}download=false`;
    }
    if (img.path) return getFileDownloadUrl(result.run_id, img.path, isDownload);
    if (img.url) {
      const sep = img.url.includes('?') ? '&' : '?';
      const base = `${API_BASE}${img.url}`;
      return isDownload ? `${base}${sep}download=true` : `${base}${sep}download=false`;
    }
    return '';
  };

  if (!result) {
    return (
      <>
        <header className="page-head">
          <div className="eyebrow">Exploratory &amp; Model Diagnostics</div>
          <h1 className="page-title">
            Generated <span className="gradient-ink">visualizations</span>
          </h1>
          <hr className="head-rule" />
        </header>
        <EmptyState
          icon="chart"
          title="No visualizations yet"
          desc="Run an autonomous pipeline to generate exploratory data analysis charts, feature correlation heatmaps, distribution plots and diagnostic confusion matrices."
        />
      </>
    );
  }

  const categories = ['ALL', ...Array.from(new Set(images.map((img) => img.category || 'GENERAL')))];
  const filteredImages = activeCategory === 'ALL' ? images : images.filter((img) => img.category === activeCategory);

  return (
    <>
      <header className="page-head" style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div className="eyebrow">Exploratory &amp; Model Diagnostics</div>
          <h1 className="page-title">
            Generated <span className="gradient-ink">visualizations</span>
          </h1>
          <p className="page-lede" style={{ fontSize: 'var(--text-md)' }}>
            {images.length} high-resolution statistical plots from <code>07_Visualizations/</code>.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={() => loadVisualizations(result.run_id)}>
          <Icon name="refresh" size={14} />
          Refresh charts
        </button>
      </header>
      <hr className="head-rule" style={{ marginTop: -8 }} />

      {/* Category filters */}
      <div className="filter-pills" role="group" aria-label="Filter by chart category">
        {categories.map((cat) => {
          const count = cat === 'ALL' ? images.length : images.filter((img) => (img.category || 'GENERAL') === cat).length;
          return (
            <button
              key={cat}
              className={`filter-pill ${activeCategory === cat ? 'active' : ''}`}
              onClick={() => setActiveCategory(cat)}
            >
              {cat} · {count}
            </button>
          );
        })}
      </div>

      {/* Grid */}
      {loading ? (
        <div className="loading-block">
          <div className="spinner" />
          <span>Loading high-resolution visual diagnostics…</span>
        </div>
      ) : filteredImages.length > 0 ? (
        <div className="viz-grid">
          {filteredImages.map((img, idx) => (
            <Card
              key={idx}
              className="viz-card"
              onClick={() => setSelectedImg(img)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') setSelectedImg(img);
              }}
              aria-label={`Open chart ${img.name}`}
            >
              <div className="viz-canvas">
                <img src={getFullImgUrl(img)} alt={img.name} loading="lazy" />
              </div>
              <div className="viz-caption">
                <span style={{ display: 'flex', alignItems: 'center', gap: 7, minWidth: 0 }}>
                  <span className="viz-cat">[{img.category || 'GENERAL'}]</span>
                  <span className="viz-name">{img.name}</span>
                </span>
                <span className="viz-zoom-hint">
                  <Icon name="search" size={13} />
                  view
                </span>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--ink-3)' }}>
            No charts found in category “{activeCategory}”.
          </div>
        </Card>
      )}

      {/* Lightbox */}
      {selectedImg &&
        createPortal(
          <div className="lightbox-overlay" onClick={() => setSelectedImg(null)}>
            <div
              className="lightbox-panel"
              onClick={(e) => e.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-label={selectedImg.name}
            >
              <div className="lightbox-head">
                <div className="lightbox-title">
                  <span style={{ color: 'var(--ember-1)' }}>[{selectedImg.category || 'GENERAL'}]</span>{' '}
                  {selectedImg.name}
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <a
                    href={getFullImgUrl(selectedImg, true)}
                    download={selectedImg.name}
                    className="btn btn-primary btn-sm"
                  >
                    <Icon name="download" size={14} />
                    PNG
                  </a>
                  <button className="btn btn-ghost btn-sm" onClick={() => setSelectedImg(null)}>
                    <Icon name="x" size={14} />
                    Close
                  </button>
                </div>
              </div>
              <div className="lightbox-canvas">
                <img src={getFullImgUrl(selectedImg)} alt={selectedImg.name} />
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
