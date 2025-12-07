"""Stage 05: MEC distribution and heterogeneity analysis."""
import argparse

from experiments.exp_mec_analysis import run_mec_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 05: MEC distribution and heterogeneity analysis."
    )
    _ = parser.parse_args()
    run_mec_analysis()


if __name__ == "__main__":
    main()
