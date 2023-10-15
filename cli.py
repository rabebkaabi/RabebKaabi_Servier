# myflaskapp/cli.py

import argparse
from myflaskapp.app import make_prediction, train_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="My Flask App CLI")
    parser.add_argument("command", help="Available commands: train, evaluate, predict")
    args = parser.parse_args()

    if args.command == "train":
        train_model()
    elif args.command == "evaluate":
        evaluate_model()
    elif args.command == "predict":
        make_prediction()
    else:
        print("Invalid command. Use 'servier train', 'servier evaluate', or 'servier predict'.")

if __name__ == "__main__":
    main()
