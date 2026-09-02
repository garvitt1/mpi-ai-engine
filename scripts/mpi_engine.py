"""
Command-line entry point for the MPI AI Engine.
"""

from pathlib import Path
import sys


# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.engine.mpi_engine import MPIEngine  # noqa: E402


def main() -> None:

    if len(sys.argv) > 1:
        requirement = " ".join(sys.argv[1:])
    else:
        requirement = input(
            "Customer requirement: "
        ).strip()

    try:

        engine = MPIEngine()

        engine.run(requirement)

    except KeyboardInterrupt:

        print("\nStopped.")

    except Exception as exc:

        print()
        print("MPI ENGINE ERROR")
        print(exc)
        print()

        sys.exit(1)


if __name__ == "__main__":
    main()