"""Build the PlaceSphere comparison using the shared filmstrip style."""

from pathlib import Path

import PullCubeTool_policy_trajectory_comparison_source as filmstrip


filmstrip.BASELINE_DIR = Path(r"C:\Users\Lenovo\Downloads\placesphere\baseline")
filmstrip.OURS_DIR = Path(r"C:\Users\Lenovo\Downloads\placesphere\our_method")
filmstrip.OUTPUT_PATH = Path(__file__).with_name(
    "PlaceSphere_policy_trajectory_comparison.png"
)


if __name__ == "__main__":
    filmstrip.main()
