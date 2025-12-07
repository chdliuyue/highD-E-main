import argparse

from experiments.exp_behavior_profiling import run_behavior_profiling_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 05: Behavioral profiling of high-interaction episodes."
    )
    _ = parser.parse_args()
    run_behavior_profiling_experiment()


if __name__ == "__main__":
    main()
