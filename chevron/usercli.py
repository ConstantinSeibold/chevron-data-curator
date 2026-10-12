"""`chevron user ...`: manage the accounts of a hosted install from the shell.

    chevron user add NAME [--admin] [--root DIR]     asks for the password twice
    chevron user passwd NAME [--root DIR]
    chevron user role NAME admin|member [--root DIR]
    chevron user remove NAME [--root DIR]
    chevron user list [--root DIR]

The first admin has to be made here; after that, admins can manage users in the launcher.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from .auth import SessionStore, UserStore


def _password(ask_twice: bool = True) -> str:
    if not sys.stdin.isatty():                       # scripted: one line on stdin
        return sys.stdin.readline().rstrip("\n")
    pw = getpass.getpass("Password: ")
    if ask_twice and getpass.getpass("Again: ") != pw:
        raise ValueError("the passwords differ")
    return pw


def main(argv: list[str]) -> int:
    from .server import DEFAULT_ROOT
    # --root is accepted before or after the subcommand (`chevron user add ana --admin --root DIR`)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS,
                        help=f"projects directory (default {DEFAULT_ROOT})")
    ap = argparse.ArgumentParser(prog="chevron user", description="manage the accounts of a hosted install",
                                 parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", parents=[common]); a.add_argument("name"); a.add_argument("--admin", action="store_true")
    sub.add_parser("passwd", parents=[common]).add_argument("name")
    r = sub.add_parser("role", parents=[common]); r.add_argument("name")
    r.add_argument("role", choices=["admin", "member"])
    sub.add_parser("remove", parents=[common]).add_argument("name")
    sub.add_parser("list", parents=[common])
    args = ap.parse_args(argv)
    root = Path(getattr(args, "root", DEFAULT_ROOT)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    users = UserStore(root)
    try:
        if args.cmd == "add":
            u = users.add(args.name, _password(), role="admin" if args.admin else "member")
            print(f"added {u.name} ({u.role})")
        elif args.cmd == "passwd":
            users.set_password(args.name, _password())
            SessionStore(root).revoke_user(args.name.lower())
            print(f"password changed, {args.name} is signed out everywhere")
        elif args.cmd == "role":
            users.set_role(args.name, args.role)
            print(f"{args.name} is now {args.role}")
        elif args.cmd == "remove":
            users.remove(args.name)
            SessionStore(root).revoke_user(args.name.lower())
            print(f"removed {args.name}")
        else:
            for u in users.list():
                print(f"{u['name']:<32} {u['role']}")
    except KeyError as e:
        print(f"no such user: {e.args[0]}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0
