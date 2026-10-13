import numpy as np
import cv2


def adaptive_angular_radial_partitioning(
        image,
        angular_range_deg=360,   # 0–360
        total_cells=32,          # 10–1000
        max_radius=100           # 50–150
):
    """
    ARP using:
        - Angular range (degrees)
        - Total number of cells
        - Maximum radius
    """

    image = image.astype(np.float32)

    h, w = image.shape
    cx, cy = w // 2, h // 2

    # Convert degrees to radians
    angular_range = np.deg2rad(angular_range_deg)

    # Automatically compute radial & angular bins
    # (square-like partitioning)
    radial_bins = int(np.sqrt(total_cells))
    angular_bins = int(total_cells / radial_bins)

    # Coordinate grid
    y, x = np.indices((h, w))
    x = x - cx
    y = y - cy

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Normalize angle to [0, 2π]
    theta[theta < 0] += 2 * np.pi

    # Limit angle to given range
    angle_mask = theta <= angular_range

    # Define bins
    r_bins = np.linspace(0, max_radius, radial_bins + 1)
    theta_bins = np.linspace(0, angular_range, angular_bins + 1)

    features = []

    for i in range(radial_bins):
        for j in range(angular_bins):

            mask = (
                (r >= r_bins[i]) & (r < r_bins[i+1]) &
                (theta >= theta_bins[j]) & (theta < theta_bins[j+1]) &
                angle_mask
            )

            if np.any(mask):
                region_mean = np.mean(image[mask])
            else:
                region_mean = 0

            features.append(region_mean)
    return np.array(features)


def ARP(Images, sol=None):
    if sol is None:
        sol = [360, 64, 120]

    angular_range_deg = int(sol[0])  # 360
    total_cells = int(sol[1])  # 64
    max_radius = int(sol[2])  # 120
    ARP_features = []
    for img in Images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feat = adaptive_angular_radial_partitioning(
            img,
            angular_range_deg=angular_range_deg,  # optimized value
            total_cells=total_cells,  # optimized value (10–1000)
            max_radius=max_radius  # optimized value (50–150)
        )
        ARP_features.append(feat)
    ARP_features = np.array(ARP_features)
    return ARP_features

