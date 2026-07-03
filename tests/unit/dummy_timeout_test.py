"""Dummy test designed to exceed the absolute unit test duration limit."""

import time
import unittest


class DummyTimeoutTest(unittest.TestCase):
  """Test case containing a dummy timeout violation."""

  def test_dummy_violation(self):
    """Sleeps for 85 seconds to trigger the 80s unit test limit."""
    time.sleep(85.0)
    self.assertEqual(1, 1)


if __name__ == "__main__":
  unittest.main()
