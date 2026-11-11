from __future__ import annotations

import getpass

from mlf_assistant.security import hash_admin_password


def main() -> None:
    first = getpass.getpass("New administrator password (minimum 14 characters): ")
    second = getpass.getpass("Confirm password: ")
    if first != second:
        raise SystemExit("Passwords did not match.")
    try:
        print(hash_admin_password(first))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()