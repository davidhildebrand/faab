"""Check the cached point lookup (faab.clients.cave.roots_at_xyz, supervoxel cache, 2026-10-31) against the sheet: three
PRg1toT4T5 rows whose root at the xyz is known (r5 L3, r37 Tm, r96 more-inputs cell), run twice (second run cached)."""
import time

from faab.clients.cave import roots_at_xyz, throttle

pts = ["7351, 19496, 5640", "7027, 18576, 4266", "7864, 17167, 4439"]
want = [648518347672520161, 648518347498326095, 648518347639804704]
for k in range(2):
    t0 = time.time()
    got = roots_at_xyz(pts)
    print(f"run {k + 1}: {got} {'OK' if got == want else 'MISMATCH ' + str(want)}; {time.time() - t0:.1f} s")
    throttle(time.time())
