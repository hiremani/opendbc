"""stamp_secoc_many signs a batch of frames in one AES call; it must match stamp_secoc byte for byte."""
import os
import random
import unittest

from opendbc.car.fisker.secoc import stamp_secoc, stamp_secoc_many


class TestSecocBatch(unittest.TestCase):
  def test_matches_single(self):
    rng = random.Random(7)
    for _ in range(20):
      key = os.urandom(16)
      can_id = rng.choice([0x1D0, 0x121])
      frames = [(bytes(rng.randrange(256) for _ in range(8)), rng.randrange(1 << 16), rng.randrange(1 << 24),
                 rng.randrange(1 << 16)) for _ in range(rng.randrange(1, 30))]
      self.assertEqual(stamp_secoc_many(key, can_id, frames), [stamp_secoc(key, can_id, *f) for f in frames])

  def test_empty(self):
    self.assertEqual(stamp_secoc_many(os.urandom(16), 0x1D0, []), [])


if __name__ == "__main__":
  unittest.main()
