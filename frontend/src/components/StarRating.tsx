import type { KeyboardEvent } from 'react'

interface Props {
  value?: number
  label?: string
  disabled?: boolean
  onChange: (rating: number) => void
}

export function StarRating({ value = 0, label = 'Đánh giá phim', disabled = false, onChange }: Props) {
  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>, star: number) {
    if (!['ArrowLeft', 'ArrowDown', 'ArrowRight', 'ArrowUp'].includes(event.key)) return
    event.preventDefault()
    const direction = event.key === 'ArrowRight' || event.key === 'ArrowUp' ? 1 : -1
    const nextRating = Math.min(5, Math.max(1, star + direction))
    onChange(nextRating)
    event.currentTarget.parentElement
      ?.querySelector<HTMLButtonElement>(`[data-rating="${nextRating}"]`)
      ?.focus()
  }

  return (
    <div className="star-rating" role="radiogroup" aria-label={`${label}. Hiện tại ${value}/5 sao`} aria-busy={disabled}>
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          type="button"
          role="radio"
          key={star}
          className={star <= value ? 'star active' : 'star'}
          data-rating={star}
          aria-label={`Chấm ${star} sao`}
          aria-checked={value === star}
          tabIndex={value === star || (value === 0 && star === 1) ? 0 : -1}
          disabled={disabled}
          onClick={() => onChange(star)}
          onKeyDown={(event) => handleKeyDown(event, star)}
        >
          <span aria-hidden="true">★</span>
        </button>
      ))}
    </div>
  )
}
