import numpy as np


class OBB:
    """
    Oriented Bounding Box.

    frame:
        4x4 homogeneous transformation matrix.
        The box is centered at the frame origin.

    size_x, size_y, size_z:
        Full side lengths.
    """

    def __init__(self, frame, size_x, size_y, size_z):
        self.frame = np.asarray(frame, dtype=float)

        if self.frame.shape != (4, 4):
            raise ValueError("frame must be a 4x4 matrix")

        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.size_z = float(size_z)

    @staticmethod
    def random_frame_from_y(origin, y_axis):
        """
        Create a random right-handed frame whose Y axis is y_axis.

        Parameters
        ----------
        origin : array-like (3,)
            Frame origin.
        y_axis : array-like (3,)
            Desired Y axis.

        Returns
        -------
        frame : (4,4) ndarray
            Homogeneous transformation matrix.
        """

        origin = np.asarray(origin, dtype=float)
        y = np.asarray(y_axis, dtype=float)

        y /= np.linalg.norm(y)

        # Random vector not parallel to y
        while True:
            v = np.random.randn(3)
            v /= np.linalg.norm(v)

            if abs(np.dot(v, y)) < 0.99:
                break

        # Build orthonormal basis
        x = np.cross(v, y)
        x /= np.linalg.norm(x)

        z = np.cross(y, x)
        z /= np.linalg.norm(z)

        frame = np.eye(4)
        frame[:3, 0] = x
        frame[:3, 1] = y
        frame[:3, 2] = z
        frame[:3, 3] = origin

        return frame
    
    @property
    def center(self):
        return self.frame[:3, 3]

    @property
    def axes(self):
        """
        Returns the three local box axes in world coordinates.
        Assumes the rotation part is orthonormal.
        """
        R = self.frame[:3, :3]
        return [R[:, 0], R[:, 1], R[:, 2]]

    @property
    def half_sizes(self):
        return np.array([
            self.size_x * 0.5,
            self.size_y * 0.5,
            self.size_z * 0.5
        ])

    def corners(self):
        """
        Returns the 8 corner points.
        """
        hx, hy, hz = self.half_sizes
        R = self.frame[:3, :3]
        c = self.center

        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = np.array([sx * hx, sy * hy, sz * hz])
                    corners.append(c + R @ local)

        return np.array(corners)

    def overlaps(self, other, eps=1e-8):
        """
        OBB/OBB intersection using SAT.
        Returns True if the boxes overlap.
        """

        A = np.column_stack(self.axes)
        B = np.column_stack(other.axes)

        a = self.half_sizes
        b = other.half_sizes

        # Rotation matrix expressing B in A coordinates
        R = A.T @ B

        # Translation in world coordinates
        t_world = other.center - self.center

        # Translation in A coordinates
        t = A.T @ t_world

        AbsR = np.abs(R) + eps

        #
        # Test A's axes
        #
        for i in range(3):
            ra = a[i]
            rb = np.dot(b, AbsR[i, :])

            if abs(t[i]) > ra + rb:
                return False

        #
        # Test B's axes
        #
        for j in range(3):
            ra = np.dot(a, AbsR[:, j])
            rb = b[j]

            if abs(np.dot(t, R[:, j])) > ra + rb:
                return False

        #
        # Test cross products A_i x B_j
        #
        for i in range(3):
            for j in range(3):

                ra = (
                    a[(i + 1) % 3] * AbsR[(i + 2) % 3, j]
                    + a[(i + 2) % 3] * AbsR[(i + 1) % 3, j]
                )

                rb = (
                    b[(j + 1) % 3] * AbsR[i, (j + 2) % 3]
                    + b[(j + 2) % 3] * AbsR[i, (j + 1) % 3]
                )

                val = abs(
                    t[(i + 2) % 3] * R[(i + 1) % 3, j]
                    - t[(i + 1) % 3] * R[(i + 2) % 3, j]
                )

                if val > ra + rb:
                    return False

        return True