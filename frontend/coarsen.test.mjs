import assert from "node:assert/strict";
import { test } from "node:test";
import {
  VIEW_GROUP_LIMIT,
  attemptsAfterLoad,
  canCoarsen,
  coarserBucket,
  initialBucket,
} from "./src/coarsen.js";

// One page opened early and left open while the run trains: groups grow as
// metrics x buckets, and every capacity error coarsens and reloads the view.
function follow({ metrics = 8, openedAt = 1000, until = 100000, every = 500 } = {}) {
  let bucket = initialBucket(openedAt);
  let attempts = 0;
  for (let steps = openedAt; steps <= until; steps += every) {
    while (metrics * Math.ceil(steps / bucket) > VIEW_GROUP_LIMIT) {
      if (!canCoarsen(attempts)) return steps; // the page stops charting here
      bucket = coarserBucket(bucket, steps);
      attempts++;
      const loaded = metrics * Math.ceil(steps / bucket);
      if (loaded <= VIEW_GROUP_LIMIT) attempts = attemptsAfterLoad(attempts, loaded);
    }
  }
  return null;
}

test("a page left open on a long run keeps charting", () => {
  assert.equal(follow(), null);
  assert.equal(follow({ metrics: 32, until: 1000000, every: 5000 }), null);
});

test("a view that cannot fit even when coarsened still gives up", () => {
  // Attempt partitions do not shrink with wider buckets: loads never regain headroom.
  let attempts = 0;
  while (canCoarsen(attempts)) attempts = attemptsAfterLoad(attempts + 1, VIEW_GROUP_LIMIT);
  assert.equal(attempts, 4);
});

test("a capacity error never leaves the bucket finer than the run length implies", () => {
  assert.equal(coarserBucket(6, 100000), initialBucket(100000));
  assert.equal(coarserBucket(1000, 100), 2000);
});
