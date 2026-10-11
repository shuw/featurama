// A challenge's progress mark: an empty ring, filled when done.
export function Ring({ full }: { full: boolean }) {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-label={full ? "Done" : "Not yet"} role="img" className="ring">
      <circle cx="14" cy="14" r="11" className="ring-track" />
      {full && <circle cx="14" cy="14" r="11" className="ring-fill" />}
    </svg>
  );
}
