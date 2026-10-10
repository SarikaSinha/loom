from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Never

import httpx

from loom.config import Config
from loom.contracts import LoomError, canonical
from loom.documents import DocumentStore
from loom.documents.embeddings import setup
from loom.trace import TraceSink


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self.print_usage(sys.stderr)
        print(json.dumps({"status": "failed", "error": "invalid_command"}))
        self.exit(1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = Parser(prog="loom")
    parser.add_argument("--version", action="version", version="loom 0.1.0")
    groups = parser.add_subparsers(dest="group", required=True)
    prepare = groups.add_parser("setup").add_subparsers(dest="operation", required=True)
    prepare.add_parser("embeddings").add_argument("--config")
    index = groups.add_parser("index").add_subparsers(dest="operation", required=True)
    for operation in ("build", "refresh"):
        command = index.add_parser(operation)
        command.add_argument("store")
        command.add_argument("--config")
    trace = groups.add_parser("trace").add_subparsers(dest="operation", required=True)
    read = trace.add_parser("read")
    read.add_argument("--config")
    read.add_argument("--run-id")
    crypto = groups.add_parser("crypto").add_subparsers(dest="operation", required=True)
    for operation in ("verify", "rekey"):
        crypto.add_parser(operation).add_argument("--config")
    relocate = crypto.add_parser("relocate")
    relocate.add_argument("--from-config", required=True)
    relocate.add_argument("--to-config", required=True)
    relocate.add_argument("--abort", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.group == "crypto" and args.operation == "relocate":
            from loom.security.relocation import relocate as move

            report = move(
                Config.load(args.from_config), Config.load(args.to_config), abort=args.abort
            )
            print(canonical(report).decode("utf-8"))
            return 0
        config = Config.load(args.config)
        if args.group == "crypto":
            from loom.security.admin import rekey, verify

            report = (rekey if args.operation == "rekey" else verify)(config)
            print(canonical(report).decode("utf-8"))
            return 0
        if args.group == "setup":
            setup(config)
            print('{"status":"completed","operation":"setup_embeddings"}')
            return 0
        if args.group == "trace":
            for event in TraceSink(config).read(args.run_id):
                print(canonical(event).decode("utf-8"))
            return 0
        binding = config.stores.get(args.store)
        if binding is None:
            raise LoomError("unknown_store")
        store = DocumentStore(path=config.path(binding.path), name=args.store).bind(config)
        report = store.build(refresh=args.operation == "refresh")
        print(canonical(report).decode("utf-8"))
        return 2 if report.get("excluded") or report.get("partially_extracted") else 0
    except KeyboardInterrupt:
        print("loom: interrupted", file=sys.stderr)
        return 130
    except (LoomError, OSError, ValueError, httpx.HTTPError) as error:
        code = error.code if isinstance(error, LoomError) else "operation_failed"
        print(f"loom: {code}", file=sys.stderr)
        print(json.dumps({"status": "failed", "error": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
