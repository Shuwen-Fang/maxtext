"""Parses JUnit XML test results and converts them to benchmark JSON format."""

import xml.etree.ElementTree as ET
import glob
import json
import sys
import os


def main():
  """Main function to parse JUnit XML files to JSON format."""
  if len(sys.argv) < 3:
    print("Usage: python parse_junit_to_benchmark.py <xml_dir> <output_json>")
    sys.exit(1)

  xml_dir = sys.argv[1]
  output_json = sys.argv[2]

  benchmarks = []
  total_times_by_device = {}

  xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
  for xml_file in xml_files:
    basename = os.path.basename(xml_file)
    # e.g., test-results-tpu-1.xml -> device = tpu
    device = "unknown"
    parts = basename.replace(".xml", "").split("-")
    if len(parts) >= 3:
      device = parts[2]

    try:
      tree = ET.parse(xml_file)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error parsing {xml_file}: {e}")
      sys.exit(1)

    root = tree.getroot()

    for testsuite in root.iter("testsuite"):
      time_val = float(testsuite.get("time", 0.0))

      # Recommend NOT adding individual file-level times to the benchmark action list
      # to prevent rebalancing/sharding from triggering false regression alerts.
      # Let Codecov handle the micro-level tracking.

      total_times_by_device[device] = total_times_by_device.get(device, 0.0) + time_val

  for device, total_time in total_times_by_device.items():
    benchmarks.append({"name": f"Total {device.upper()} Tests Duration", "unit": "sec", "value": total_time})

  with open(output_json, "w", encoding="utf-8") as f:
    json.dump(benchmarks, f, indent=2)

  print(f"Parsed {len(xml_files)} XML files and extracted {len(benchmarks)} duration metrics.")


if __name__ == "__main__":
  main()
