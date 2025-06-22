# cli.py

import argparse
import subprocess
import sys
from forestfire_sim import predict, simulate

def run_dashboard():
    """Launch the dashboard app (Streamlit-based)."""
    print("Starting Streamlit Dashboard...")
    subprocess.run(["streamlit", "run", "app/app.py"])

def run_prediction(area):
    """Run the fire risk prediction for a specified area."""
    print(f"Running fire risk prediction for {area}...")
    predict.run_model(area)  # Example callable

def run_simulation(area, hours):
    """Run fire spread simulation."""
    print(f"Simulating fire spread for {area} over {hours} hours...")
    simulate.run_simulation(area, hours)

def main():
    parser = argparse.ArgumentParser(description="Forest Fire Simulation CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Dashboard
    parser_dashboard = subparsers.add_parser("dashboard", help="Launch the dashboard UI")

    # Predict
    parser_predict = subparsers.add_parser("predict", help="Run fire risk prediction")
    parser_predict.add_argument("--area", type=str, required=True, help="Target area (e.g., uttarakhand)")

    # Simulate
    parser_simulate = subparsers.add_parser("simulate", help="Run fire spread simulation")
    parser_simulate.add_argument("--area", type=str, required=True, help="Target area")
    parser_simulate.add_argument("--hours", type=int, default=3, help="Number of hours to simulate")

    args = parser.parse_args()

    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "predict":
        run_prediction(args.area)
    elif args.command == "simulate":
        run_simulation(args.area, args.hours)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
