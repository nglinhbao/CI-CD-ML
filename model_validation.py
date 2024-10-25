# model_validation.py
import sys


def read_metrics(filename):
    """Read RMSE and R2 from metrics file."""
    with open(filename, "r") as f:
        content = f.read()
        rmse = float(content.split("Root Mean Squared Error = ")[1].split(",")[0])
        r2 = float(content.split("R-squared Score = ")[1])
    return rmse, r2


def compare_models():
    """Compare new model performance with previous version."""
    try:
        # Read previous metrics
        prev_rmse, prev_r2 = read_metrics("Results/previous_metrics.txt")

        # Read new metrics
        new_rmse, new_r2 = read_metrics("Results/metrics.txt")

        # Compare performance
        rmse_improved = new_rmse < prev_rmse
        r2_improved = new_r2 > prev_r2

        # Print comparison
        print("\nModel Performance Comparison:")
        print(f"Previous RMSE: {prev_rmse:.4f}, New RMSE: {new_rmse:.4f}")
        print(f"Previous R2: {prev_r2:.4f}, New R2: {new_r2:.4f}")

        # Determine if new model is better
        if rmse_improved and r2_improved:
            print("\nNew model shows improvement in both metrics!")
            return 0
        elif rmse_improved or r2_improved:
            print("\nNew model shows mixed results - manual review recommended")
            return 0
        else:
            print("\nNew model performs worse than previous version")
            return 1

    except FileNotFoundError:
        print("\nNo previous metrics found - this is the first model version")
        return 0


if __name__ == "__main__":
    sys.exit(compare_models())
