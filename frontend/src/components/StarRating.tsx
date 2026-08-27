interface Props {
  value?: number
  disabled?: boolean
  onChange: (rating: number) => void
}

export function StarRating({ value = 0, disabled = false, onChange }: Props) {
  return (
    <div className="star-rating" aria-label={`Đánh giá ${value}/5`}>
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          type="button"
          key={star}
          className={star <= value ? 'star active' : 'star'}
          aria-label={`${star} sao`}
          disabled={disabled}
          onClick={() => onChange(star)}
        >
          ★
        </button>
      ))}
    </div>
  )
}
