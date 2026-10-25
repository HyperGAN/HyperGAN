// Bucket policy for a live view whose chart state grows with the run.
// view-worker.js MAX_GROUPS: live chart groups the browser keeps per view.
export const VIEW_GROUP_LIMIT = 2048;
// Capacity errors in a row, without regaining headroom, before a view gives up.
export const MAX_COARSENINGS = 4;
const TARGET_BUCKETS = 180;

export function initialBucket(steps) {
  return Math.max(1, Math.ceil((steps || 1) / TARGET_BUCKETS));
}

// After a capacity error: at least double, and never finer than the run's
// current length implies.
export function coarserBucket(bucket, steps) {
  return Math.max(bucket * 2, initialBucket(steps));
}

// A view that loaded with room to spare starts counting afresh, so a page left
// open on a long run keeps widening its buckets as the run grows. Doubling at
// the limit leaves just over half of it in use; a quarter of headroom means the
// next capacity error needs real growth, so this can never loop.
export function attemptsAfterLoad(attempts, groups) {
  return groups <= (VIEW_GROUP_LIMIT * 3) / 4 ? 0 : attempts;
}

export function canCoarsen(attempts) {
  return attempts < MAX_COARSENINGS;
}
