import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Card } from '../components/Card';
import { listVisualizations, getFileDownloadUrl, API_BASE } from '../api';

export function Visualizations({ result }) {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeCategory, setActiveCategory] = useState('ALL');
  const [selectedImg, setSelectedImg] = useState(null);

  useEffect(() => {
    if (result && result.run_id) {
      loadVisualizations(result.run_id);
    }
  }, [result]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSelectedImg(null);
      }
    };
    if (selectedImg) {
      window.addEventListener('keydown', handleKeyDown);
    }
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
      <div className="empty-state">
        <div className="empty-state-icon">📈</div>
        <div className="empty-state-title">No Visualizations Available</div>
        <div className="empty-state-desc">
          Run an autonomous pipeline to automatically generate exploratory data analysis charts, feature correlation heatmaps, distribution plots, and diagnostic confusion matrices.
        </div>
      </div>
    );
  }

  // Extract unique categories
  const categories = ['ALL', ...Array.from(new Set(images.map((img) => img.category || 'GENERAL')))];

  const filteredImages = activeCategory === 'ALL'
    ? images
    : images.filter((img) => img.category === activeCategory);

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: '28px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div className="eyebrow">Exploratory & Model Diagnostics</div>
          <h2 style={{ margin: '4px 0 8px 0' }}>
            Generated Visualizations ({images.length} charts)
          </h2>
          <p style={{ margin: 0, color: 'var(--text-secondary)' }}>
            High-resolution statistical plots, correlation matrices, feature distributions, and model evaluation heatmaps stored in <code>07_Visualizations/</code>.
          </p>
        </div>

        {images.length > 0 && (
          <button
            className="btn btn-ghost"
            style={{ fontSize: 'var(--text-xs)' }}
            onClick={() => loadVisualizations(result.run_id)}
          >
            🔄 Refresh Charts
          </button>
        )}
      </div>

      {/* Category Filter Pills */}
      {images.length > 0 && (
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '24px' }}>
          {categories.map((cat) => {
            const count = cat === 'ALL' ? images.length : images.filter((img) => img.category === cat).length;
            const isActive = activeCategory === cat;
            return (
              <button
                key={cat}
                className={`btn ${isActive ? 'btn-primary' : 'btn-ghost'}`}
                style={{ fontSize: 'var(--text-xs)', padding: '6px 14px', borderRadius: 'var(--radius-full)' }}
                onClick={() => setActiveCategory(cat)}
              >
                {cat} ({count})
              </button>
            );
          })}
        </div>
      )}

      {/* Image Grid */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '60px', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: '2rem', marginBottom: '12px' }}>⏳</div>
          Loading high-resolution visual diagnostics...
        </div>
      ) : filteredImages.length > 0 ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px' }}>
          {filteredImages.map((img, idx) => {
            const imgUrl = getFullImgUrl(img);
            return (
              <Card
                key={idx}
                style={{
                  padding: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  background: 'var(--surface-2)',
                  cursor: 'pointer',
                  transition: 'transform 0.2s var(--ease-spring), border-color 0.2s var(--ease-out-expo)',
                }}
                onClick={() => setSelectedImg(img)}
              >
                {/* Plot Canvas Container with crisp contrast */}
                <div
                  style={{
                    background: '#ffffff',
                    borderRadius: 'var(--radius-md)',
                    padding: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '260px',
                    overflow: 'hidden',
                  }}
                >
                  <img
                    src={imgUrl}
                    alt={img.name}
                    loading="lazy"
                    style={{
                      maxWidth: '100%',
                      height: 'auto',
                      display: 'block',
                      objectFit: 'contain',
                    }}
                  />
                </div>

                {/* Caption Bar */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '10px', padding: '0 4px' }}>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                    <span style={{ color: 'var(--accent-indigo)', fontWeight: 600 }}>[{img.category}]</span> {img.name}
                  </div>
                  <span style={{ fontSize: 'var(--text-xs)', color: '#d8b4fe', fontWeight: 600 }}>🔍 Click to view full image</span>
                </div>
              </Card>
            );
          })}
        </div>
      ) : (
        <Card>
          <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
            No charts found matching category "{activeCategory}".
          </div>
        </Card>
      )}

      {/* Lightbox Fullscreen Modal via React Portal */}
      {selectedImg && createPortal(
        <div
          className="lightbox-overlay"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(3, 1, 10, 0.88)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            zIndex: 999999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px',
            animation: 'lightboxFadeIn 0.25s ease-out forwards',
          }}
          onClick={() => setSelectedImg(null)}
        >
          <div
            className="lightbox-content"
            style={{
              maxWidth: '92vw',
              maxHeight: '92vh',
              background: 'var(--surface-2)',
              border: '1px solid var(--border-medium)',
              borderRadius: 'var(--radius-xl)',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 24px 60px rgba(0, 0, 0, 0.85), 0 0 30px rgba(147, 51, 234, 0.3)',
              animation: 'lightboxZoomIn 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
              <div style={{ fontWeight: 700, fontSize: 'var(--text-md)', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                <span style={{ color: 'var(--accent-indigo)' }}>[{selectedImg.category}]</span> {selectedImg.name}
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <a
                  href={getFullImgUrl(selectedImg, true)}
                  download={selectedImg.name}
                  className="btn btn-primary"
                  style={{ fontSize: 'var(--text-xs)', padding: '6px 14px' }}
                >
                  📥 Download High-Res PNG
                </a>
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: 'var(--text-xs)', padding: '6px 14px' }}
                  onClick={() => setSelectedImg(null)}
                >
                  ✕ Close
                </button>
              </div>
            </div>

            <div style={{ background: '#ffffff', borderRadius: 'var(--radius-md)', padding: '16px', overflow: 'auto', textAlign: 'center', boxShadow: 'inset 0 0 10px rgba(0,0,0,0.05)' }}>
              <img
                src={getFullImgUrl(selectedImg)}
                alt={selectedImg.name}
                style={{ maxWidth: '100%', maxHeight: '74vh', objectFit: 'contain', display: 'inline-block' }}
              />
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
