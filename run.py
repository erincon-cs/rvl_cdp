import argparse

import json
from server import app


def main():
    global classifier

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-path")
    arg_parser.add_argument("--host", default="0.0.0.0", type=str)
    arg_parser.add_argument("--port", default=5000, type=int)
    arg_parser.add_argument("--dataset")
    args = arg_parser.parse_args()

    data = {
        "model_path": args.model_path,
        "dataset": args.dataset
    }

    with open('server/args.json', 'w') as outfile:
        json.dump(data, outfile)

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
