import React, { useState, useRef, useEffect } from 'react';

export function SearchableSelect({
  value = '',
  onChange,
  options = [],
  placeholder = 'Type to search or select model...',
  direction = 'up', // 'up' | 'down'
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState(value || '');
  const containerRef = useRef(null);

  useEffect(() => {
    setQuery(value || '');
  }, [value]);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const filteredOptions = query
    ? options.filter((opt) => opt.toLowerCase().includes(query.toLowerCase()))
    : options;

  const handleSelect = (val) => {
    setQuery(val);
    if (onChange) onChange(val);
    setIsOpen(false);
  };

  const handleInputChange = (e) => {
    const val = e.target.value;
    setQuery(val);
    if (onChange) onChange(val);
    setIsOpen(true);
  };

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%' }}>
      <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
        <input
          type="text"
          className="settings-input"
          value={query}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          placeholder={placeholder}
          style={{
            width: '100%',
            paddingRight: '36px',
            fontFamily: 'var(--font-mono)',
            fontSize: 'var(--text-xs)',
          }}
        />
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          style={{
            position: 'absolute',
            right: '8px',
            background: 'none',
            border: 'none',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            padding: '6px',
            display: 'flex',
            alignItems: 'center',
            fontSize: '12px',
          }}
          title={isOpen ? 'Close menu' : 'Open model list'}
        >
          {isOpen ? '▲' : '▼'}
        </button>
      </div>

      {isOpen && (
        <div
          style={{
            position: 'absolute',
            bottom: direction === 'up' ? 'calc(100% + 4px)' : 'auto',
            top: direction === 'down' ? 'calc(100% + 4px)' : 'auto',
            left: 0,
            right: 0,
            maxHeight: '260px',
            overflowY: 'auto',
            background: '#121316',
            border: '1px solid var(--border-medium)',
            borderRadius: 'var(--radius-md)',
            boxShadow: '0 8px 30px rgba(0, 0, 0, 0.7)',
            zIndex: 1000,
          }}
        >
          {filteredOptions.length > 0 ? (
            filteredOptions.map((opt) => {
              const isSelected = opt === value;
              return (
                <div
                  key={opt}
                  onClick={() => handleSelect(opt)}
                  style={{
                    padding: '8px 12px',
                    fontSize: 'var(--text-xs)',
                    fontFamily: 'var(--font-mono)',
                    color: isSelected ? '#38bdf8' : '#e4e4e7',
                    background: isSelected ? 'rgba(56, 189, 248, 0.15)' : 'transparent',
                    cursor: 'pointer',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <span style={{ wordBreak: 'break-all' }}>{opt}</span>
                  {isSelected && <span style={{ color: '#38bdf8', fontWeight: 'bold', marginLeft: '8px' }}>✓</span>}
                </div>
              );
            })
          ) : (
            <div style={{ padding: '12px', fontSize: 'var(--text-xs)', color: 'var(--text-muted)', textAlign: 'center' }}>
              No pre-listed match. Custom model: "<strong>{query}</strong>"
            </div>
          )}
        </div>
      )}
    </div>
  );
}
