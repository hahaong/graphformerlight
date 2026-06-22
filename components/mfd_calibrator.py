import numpy as np


class DynamicMFDCalibrator:
    """
    MODULE 1: ITERATIVE MFD CALIBRATOR
    Tracks macro-level network accumulation and trip completion rate data pairs
    during an active episode to dynamically reconstruct the network's MFD
    and update the critical accumulation threshold (n_c) for the next episode.
    """

    def __init__(self, sample_interval_secs: int = 60, initial_n_c: float = 250.0):
        self.sample_interval = sample_interval_secs
        self.current_n_c = initial_n_c

        # Episodic data tracking arrays
        self.episode_accumulation_records = []
        self.episode_tc_records = []

        print(f"[MFD Calibrator] Initialized with static baseline n_c = {self.current_n_c}")

    def collect_step_data(self, current_seconds: int, accumulation: float, trip_completion_rate: float):
        """
        To be called inside the simulation step. Collects macroscopic data points
        at the specified sampling interval (e.g., every 60 seconds).
        """
        if current_seconds % self.sample_interval == 0:
            self.episode_accumulation_records.append(accumulation)
            self.episode_tc_records.append(trip_completion_rate)

    def execute_episodic_calibration(self, episode_idx: int) -> float:
        """
        To be called at the exact termination of an episode. Fits the cubic MFD function
        via OLS and analytically solves dG/dn = 0 to update the critical accumulation.

        Returns:
            float: The newly calibrated critical accumulation (n_c) for the next episode.
        """
        n = np.array(self.episode_accumulation_records, dtype=np.float64)
        G = np.array(self.episode_tc_records, dtype=np.float64)

        # Flush metrics immediately to prepare for the subsequent episode
        self.episode_accumulation_records.clear()
        self.episode_tc_records.clear()

        # Guard rail: Ensure enough distinct data samples exist to perform regression safely
        if len(n) < 5 or np.all(n == 0):
            print(
                f"[MFD Calibrator] Episode {episode_idx}: Insufficient data points. Maintaining n_c = {self.current_n_c:.2f}")
            return self.current_n_c

        # Construct the OLS Design Matrix for: G(n) = a*n^3 + b*n^2 + c*n
        # No intercept row is added because when n=0, trip completion flow must equal 0.
        X = np.vstack([n ** 3, n ** 2, n]).T

        try:
            # Solve normal equations via OLS: theta = (X^T * X)^(-1) * X^T * G
            X_T_X_inv = np.linalg.inv(X.T @ X)
            theta = X_T_X_inv @ X.T @ G
            a, b, c = theta[0], theta[1], theta[2]

            # Analytical Peak Finding:
            # dG/dn = 3*a*n^2 + 2*b*n + c = 0
            # Standard quadratic roots formulation: n = (-2b +/- sqrt(4b^2 - 12ac)) / (6a)
            discriminant = (2.0 * b) ** 2 - 12.0 * a * c

            if discriminant >= 0 and a < 0:  # a must be negative for a downward-opening parabolic peak
                calculated_n_c = (-2.0 * b - np.sqrt(discriminant)) / (6.0 * a)

                # Physical validation bounds check
                if 10.0 < calculated_n_c < np.max(n) * 1.5:
                    self.current_n_c = float(calculated_n_c)
                    print(
                        f"[MFD Calibrator] Episode {episode_idx} SUCCESS! Fitted Coefficients: a={a:.2e}, b={b:.2e}, c={c:.2e}")
                    print(
                        f"[MFD Calibrator] Calibrated critical accumulation for next episode: n_c = {self.current_n_c:.2f}")
                else:
                    print(
                        f"[MFD Calibrator] Episode {episode_idx} WARNING: Calculated peak {calculated_n_c:.2f} out of real bounds. Retaining previous n_c.")
            else:
                print(
                    f"[MFD Calibrator] Episode {episode_idx} WARNING: Invalid MFD shape (discriminant < 0 or positive a). Retaining previous n_c.")

        except (np.linalg.LinAlgError, ValueError) as e:
            print(
                f"[MFD Calibrator] Episode {episode_idx} ERROR: Matrix inversion or numerical variance error during OLS regression: {e}")
            # Maintain previous operational state on matrix errors
            pass

        return self.current_n_c