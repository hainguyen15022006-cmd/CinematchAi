interface Props {
  value?: number
  disabled?: boolean
  onChange: (rating: number) => void
}

export function StarRating({ value = 0, disabled = false, onChange }: Props) {
  return (
    <div className="star-rating" aria-label={`Rating ${value}/5`}>
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          type="button"
          key={star}
          className={star <= value ? 'star active' : 'star'}
          aria-label={`${star} ${star === 1 ? 'star' : 'stars'}`}
          disabled={disabled}
          onClick={() => onChange(star)}
        >
          ★
        </button>
      ))}
    </div>
  )
}
