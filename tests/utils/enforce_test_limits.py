"""Enforces absolute maximum duration limits for unit and integration tests."""

import xml.etree.ElementTree as ET
import glob
import os
import sys

UNIT_TEST_LIMIT_SEC = 80.0
TRAIN_COMPILE_TEST_LIMIT_SEC = 1500.0
INTEGRATION_TEST_LIMIT_SEC = 300.0


def main():
  """Parses XML files and fails if any test exceeds the limit."""
  if len(sys.argv) < 2:
    print("Usage: python enforce_test_limits.py <xml_dir>")
    sys.exit(1)

  xml_dir = sys.argv[1]
  xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

  if not xml_files:
    print(f"No XML files found in {xml_dir}")
    sys.exit(0)

  failed = False

  for xml_file in xml_files:
    try:
      tree = ET.parse(xml_file)
      root = tree.getroot()
      for testcase in root.iter("testcase"):
        time_val = float(testcase.get("time", 0.0))
        name = testcase.get("name", "unknown")
        classname = testcase.get("classname", "unknown")
        full_name = f"{classname}.{name}"

        is_integration = (
            "integration" in xml_file.lower()
            or "tests/integration/" in full_name.lower()
            or ".integration" in full_name.lower()
        )
        is_train_compile = "train_compile" in full_name.lower() or "train_compile" in xml_file.lower()

        if is_integration:
          limit = INTEGRATION_TEST_LIMIT_SEC
          test_type = "Integration Test"
        elif is_train_compile:
          limit = TRAIN_COMPILE_TEST_LIMIT_SEC
          test_type = "Train Compile Test"
        else:
          limit = UNIT_TEST_LIMIT_SEC
          test_type = "Unit Test"

        if time_val > limit:
          print(f"::error::[PERFORMANCE ALERT] {test_type} exceeded absolute limit!")
          print(f"  Test: {full_name}")
          print(f"  File: {os.path.basename(xml_file)}")
          print(f"  Duration: {time_val:.2f}s (Limit: {limit:.2f}s)")
          print("-" * 50)
          failed = True

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error parsing {xml_file}: {e}")
      failed = True

  if failed:
    print("\nOne or more tests exceeded the absolute execution time limits.")
    sys.exit(1)
  else:
    print("All tests passed absolute execution time limits.")


if __name__ == "__main__":
  main()
