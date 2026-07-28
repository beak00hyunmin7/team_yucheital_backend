from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    size = 420
    yy, xx = np.indices((size, size))
    base_slope = 0.55 * (1.0 - xx / (size - 1)) + 0.35 * (1.0 - yy / (size - 1))
    hill = 0.32 * np.exp(-((xx - 105) ** 2 + (yy - 95) ** 2) / (2 * 62**2))
    depression = -0.28 * np.exp(-((xx - 310) ** 2 + (yy - 305) ** 2) / (2 * 54**2))
    terrain = base_slope + hill + depression
    terrain = (terrain - terrain.min()) / np.ptp(terrain)

    output_path = Path(__file__).resolve().parents[1] / "samples" / "sample_dem.png"
    Image.fromarray((terrain * 255).astype(np.uint8), mode="L").save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()

