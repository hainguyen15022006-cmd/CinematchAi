import type { ReactNode } from 'react'

export function Alert({ type, children }: { type: 'error' | 'success' | 'info'; children: ReactNode }) {
  return <div className={`alert alert-${type}`}>{children}</div>
}
