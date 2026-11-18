import React from 'react';

export function Card({ children, className = '', style, glow, jade, ...rest }) {
  const cls = ['card'];
  if (glow === 'ember') cls.push('card-glow');
  if (jade) cls.push('card-jade');
  if (className) cls.push(className);
  return (
    <div className={cls.join(' ')} style={style} {...rest}>
      {children}
    </div>
  );
}
