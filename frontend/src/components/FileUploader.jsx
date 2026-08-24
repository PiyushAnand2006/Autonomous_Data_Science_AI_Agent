import React, { useState } from 'react';
import { uploadDataset, createSampleDataset } from '../api';

export function FileUploader({ onFileReady, currentFile }) {
  const [mode, setMode] = useState('upload'); // 'upload' | 'sample'
  const [selectedSample, setSelectedSample] = useState('churn');
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null); // { percent, loaded, total }
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = async (file) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setUploadProgress({ percent: 0, loaded: 0, total: file.size });
    try {
      const res = await uploadDataset(file, (p) => {
        setUploadProgress(p);
      });
      onFileReady({ path: res.path, filename: res.filename, size: res.size_bytes });
      setUploadProgress(null);
    } catch (err) {
      setError(err.message || 'Upload failed');
      setUploadProgress(null);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleUseSample = async (sampleType = 'churn') => {
    setLoading(true);
    setError(null);
    try {
      const res = await createSampleDataset();
      onFileReady({ path: res.path, filename: res.filename, rows: res.rows, features: res.features, sampleType });
    } catch (err) {
      setError(err.message || 'Failed to load sample');
    } finally {
      setLoading(false);
    }
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setError(null);
    if (newMode === 'upload') {
      // Switching to upload clears previous sample preview
      onFileReady(null);
    } else {
      // Switching to sample loads the selected sample
      handleUseSample(selectedSample);
    }
  };

  return (
    <div>
      {/* Radio Selector: Select Dataset Mode */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: '8px', fontWeight: 500 }}>
          Select Dataset Mode
        </div>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: 'var(--text-sm)', color: mode === 'upload' ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
            <input
              type="radio"
              name="dataset_mode"
              checked={mode === 'upload'}
              onChange={() => handleModeChange('upload')}
              style={{ accentColor: '#2563eb' }}
            />
            <span>Upload My Dataset</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: 'var(--text-sm)', color: mode === 'sample' ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
            <input
              type="radio"
              name="dataset_mode"
              checked={mode === 'sample'}
              onChange={() => handleModeChange('sample')}
              style={{ accentColor: '#2563eb' }}
            />
            <span>Use Sample Dataset</span>
          </label>
        </div>
      </div>

      {mode === 'upload' ? (
        <div>
          <div
            className={`file-uploader ${dragOver ? 'dragover' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            style={{ padding: '24px 16px', textAlign: 'center', cursor: 'pointer' }}
          >
            <input
              type="file"
              accept=".csv,.xlsx,.parquet"
              onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
              disabled={loading}
            />
            <div className="file-uploader-icon" style={{ fontSize: '2rem', marginBottom: '8px' }}>
              {loading ? '⏳' : '📁'}
            </div>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
              {loading
                ? (uploadProgress ? `Uploading: ${uploadProgress.percent}% (${(uploadProgress.loaded / 1048576).toFixed(1)} MB / ${(uploadProgress.total / 1048576).toFixed(1)} MB)` : 'Uploading & validating...')
                : 'Upload CSV, Excel, or Parquet file'}
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
              500MB per file • CSV, XLSX, PARQUET
            </div>

            {uploadProgress && (
              <div style={{ width: '100%', maxWidth: '320px', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', margin: '12px auto 0 auto', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${uploadProgress.percent}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #7c3aed, #a855f7)',
                    transition: 'width 0.2s ease',
                  }}
                />
              </div>
            )}
          </div>

          {currentFile && !loading && (
            <div style={{ marginTop: '12px', padding: '10px 14px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: 'var(--radius-sm)', fontSize: 'var(--text-xs)', color: '#6ee7b7', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>✓ Ready on Backend: <strong>{currentFile.filename}</strong></span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>
                {currentFile.size ? `${(currentFile.size / 1024).toFixed(1)} KB` : ''}
              </span>
            </div>
          )}
        </div>
      ) : (
        <div>
          <div style={{ marginBottom: '12px' }}>
            <label className="settings-label" style={{ marginBottom: '6px' }}>Sample Dataset</label>
            <select
              className="settings-select"
              value={selectedSample}
              onChange={(e) => {
                const s = e.target.value;
                setSelectedSample(s);
                handleUseSample(s);
              }}
            >
              <option value="churn">Customer Churn (Classification)</option>
            </select>
          </div>

          <div style={{ padding: '12px 14px', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.25)', borderRadius: 'var(--radius-sm)', fontSize: 'var(--text-xs)', color: '#93c5fd', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>ℹ️</span>
            <span>
              Loaded <code>sample_churn.csv</code> (500 rows, 9 features)
            </span>
          </div>
        </div>
      )}

      {error && (
        <div style={{ color: 'var(--accent-rose)', fontSize: 'var(--text-xs)', marginTop: '8px', fontFamily: 'var(--font-mono)' }}>
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}
