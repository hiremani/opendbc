"""The parser's one-shift signal extraction (parser.fast_extractor) must give exactly what get_raw_value gives."""
import glob
import os
import random
import unittest

from opendbc.can.dbc import DBC
from opendbc.can.parser import MessageState, get_raw_value

DBC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dbc")


def fast_value(state: MessageState, i: int, dat: bytes) -> int:
  """The value parse() computes for signal i, before sign and scaling."""
  little, shift, mask, need = state.fast[i]
  n = len(dat)
  if need > n:
    return get_raw_value(dat, state.signals[i])
  if little:
    return (int.from_bytes(dat, "little") >> shift) & mask
  return (int.from_bytes(dat, "big") >> ((n - 1) * 8 + shift)) & mask


class TestFastExtract(unittest.TestCase):
  def test_every_signal_in_every_dbc(self):
    rng = random.Random(0)
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DBC_DIR, "*.dbc")))
    self.assertGreater(len(names), 20)
    checked = 0
    for name in names:
      try:
        dbc = DBC(name)
      except Exception:
        continue
      for msg in dbc.msgs.values() if isinstance(dbc.msgs, dict) else dbc.msgs:
        state = MessageState(address=msg.address, name=msg.name, size=msg.size, signals=list(msg.sigs.values()))
        for size in {msg.size, max(1, msg.size - 1)}:
          for _ in range(3):
            dat = bytes(rng.getrandbits(8) for _ in range(size))
            for i, sig in enumerate(state.signals):
              self.assertEqual(fast_value(state, i, dat), get_raw_value(dat, sig), f"{name} {msg.name} {sig.name} {size}B")
              checked += 1
    self.assertGreater(checked, 100_000)

  def test_parse_matches_the_byte_walk(self):
    """parse() itself, end to end: every signal's scaled value as the byte-by-byte walk computes it."""
    rng = random.Random(1)
    for name in ("fisker_ocean_adas", "fisker_ocean_mrr", "toyota_nodsu_pt_generated", "hyundai_kia_generic"):
      try:
        dbc = DBC(name)
      except Exception:
        continue
      for msg in dbc.msgs.values():
        state = MessageState(address=msg.address, name=msg.name, size=msg.size, signals=list(msg.sigs.values()),
                             ignore_checksum=True, ignore_counter=True)
        for k in range(4):
          dat = bytes(rng.getrandbits(8) for _ in range(msg.size))
          self.assertTrue(state.parse(1000 + k, dat))
          for i, sig in enumerate(state.signals):
            raw = get_raw_value(dat, sig)
            if sig.is_signed:
              raw -= ((raw >> (sig.size - 1)) & 0x1) * (1 << sig.size)
            self.assertEqual(state.vals[i], raw * sig.factor + sig.offset, f"{name} {msg.name} {sig.name}")

  def test_all_ones_and_zeros(self):
    dbc = DBC("fisker_ocean_adas")
    for msg in dbc.msgs.values():
      state = MessageState(address=msg.address, name=msg.name, size=msg.size, signals=list(msg.sigs.values()))
      for dat in (bytes(msg.size), b"\xff" * msg.size):
        for i, sig in enumerate(state.signals):
          self.assertEqual(fast_value(state, i, dat), get_raw_value(dat, sig))


if __name__ == "__main__":
  unittest.main()
