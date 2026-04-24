"""Startup banner for r3LAY services."""

from __future__ import annotations

# ANSI color codes
_ORG = "\033[38;5;208m"
_DIM = "\033[2m"
_BLD = "\033[1m"
_RST = "\033[0m"


def print_bridge_banner(port: int = 8765) -> None:
    """Print the bridge startup banner -- box with brain inline right."""
    o, d, b, r = _ORG, _DIM, _BLD, _RST
    print(
        f"\n"
        f"{o}{b} ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓{r}"
        f"{o}    _.--'\"'.{r}\n"
        f"{o}{b} ┃  r ³ L A Y                      ┃{r}"
        f"{o}   (  ( (   ){r}\n"
        f"{o}{b} ┃  local-first project brain      ┃{r}"
        f"{o}   (o)_    ) ){r}\n"
        f"{o}{b} ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛{r}"
        f"{o}       (o)_.'{r}\n"
        f"{o}  Bridge{r} {d}listening on{r} :{port}"
        f"{o}              )/{r}\n"
        f"{d}  ─────────────────────────────────{r}\n"
    )


def print_watcher_banner(watch_path: str = "~/r3LAY/") -> None:
    """Print the watcher banner -- r3LAY title with eye inline right."""
    o, d, b, r = _ORG, _DIM, _BLD, _RST
    pad = watch_path.ljust(26)
    print(
        f"\n"
        f"                                          {d}_.._{r}\n"
        f"{o}{b}  r ³ L A Y  {r}{o}Watcher{r}"
        f"                  {d}.' .-. '.{r}\n"
        f"{o}  {d}watching{r} {pad}"
        f"{d}(  ( o )  ){r}\n"
        f"{d}  ─────────────────────────────────{r}"
        f" {d}`._'-'_.'{r}\n"
        f"                                       {d}```{r}\n"
    )


def print_full_banner(port: int = 8765) -> None:
    """Print the full logo (for interactive sessions)."""
    print_bridge_banner(port)


if __name__ == "__main__":
    print_full_banner()
