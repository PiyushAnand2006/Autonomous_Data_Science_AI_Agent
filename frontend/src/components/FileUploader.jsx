import React, { useState } from 'react';
import { uploadDataset, createSampleDataset } from '../api';
import { Icon } from './Icon';

export function FileUploader({ onFileReady, currentFile }) {
  const [mode, setMode] = useState('upload'); // 'upload' | 'sample'
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
      const res = await uploadDataset(file, (p) => setUploadProgress(p));
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

  const handleUseSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await createSampleDataset();
      onFileReady({ path: res.path, filename: res.filename, rows: res.rows, features: res.features, sampleType: 'churn' });
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
      // Switching to upload clears the previous sample preview
      onFileReady(null);
    } else {
      handleUseSample();
    }
  };

  return (
    <div>
      <div className="field-label" style={{ marginBottom: 10 }}>Dataset source</div>

      <div className="segmented" role="tablist" aria-label="Dataset source">
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'upload'}
          className={`segmented-btn ${mode === 'upload' ? 'active' : ''}`}
          onClick={() => handleModeChange('upload')}
        >
          <Icon name="upload" size={15} />
          Upload my dataset
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'sample'}
          className={`segmented-btn ${mode === 'sample' ? 'active' : ''}`}
          onClick={() => handleModeChange('sample')}
        >
          <Icon name="database" size={15} />
          Use sample
        </button>
      </div>

      <div style={{ height: 14 }} />

      {mode === 'upload' ? (
        <div>
          <div
            className={`dropzone ${dragOver ? 'dragover' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".csv,.xlsx,.parquet"
              onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
              disabled={loading}
              aria-label="Upload CSV, Excel or Parquet dataset"
            />
            <div className="dropzone-icon">
              <Icon name={loading ? 'cpu' : 'upload'} size={22} />
            </div>
            <div className="dropzone-title">
              {loading && uploadProgress
                ? `Uploading ${uploadProgress.percent}% · ${(uploadProgress.loaded / 1048576).toFixed(1)} / ${(uploadProgress.total / 1048576).toFixed(1)} MB`
                : loading
                ? 'Uploading & validating…'
                : 'Drop a CSV, Excel or Parquet file'}
            </div>
            <div className="dropzone-sub">or click to browse · max 500 MB</div>

            {uploadProgress && (
              <div className="progress-track" style={{ marginTop: 12 }}>
                <div className="progress-fill" style={{ width: `${uploadProgress.percent}%` }} />
              </div>
            )}
          </div>

          {currentFile && !loading && (
            <div className="ready-chip">
              <span>
                Ready on backend: <strong>{currentFile.filename}</strong>
              </span>
              <span className="mono" style={{ fontSize: 11 }}>
                {currentFile.size ? `${(currentFile.size / 1024).toFixed(1)} KB` : ''}
              </span>
            </div>
          )}
        </div>
      ) : (
        <div className="notice notice-sand">
          <Icon name="info" size={16} />
          <span>
            Sample loaded: <code>sample_churn.csv</code> — 500 rows × 9 features (customer churn, classification).
          </span>
        </div>
      )}

      {error && (
        <div className="notice notice-alert" style={{ marginTop: 12 }}>
          <Icon name="alert-triangle" size={16} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
