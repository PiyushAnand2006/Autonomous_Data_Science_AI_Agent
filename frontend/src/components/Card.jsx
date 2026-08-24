import React from 'react';

export function Card({ children, glow, className = '', ...props }) {
  const glowClass = glow === 'emerald' ? 'glow-emerald' : glow === 'sapphire' ? 'glow-sapphire' : '';
  return (
    <div className={`card-shell ${glowClass} ${className}`} {...props}>
      <div className="card-core">
        {children}
      </div>
    </div>
  );
}
