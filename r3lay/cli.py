"""CLI commands for r3LAY garage terminal.

Provides subcommands for maintenance logging and mileage tracking.

Commands:
    r3lay log oil       - Log an oil change
    r3lay log service   - Log a scheduled service
    r3lay log repair    - Log a repair
    r3lay log mod       - Log a modification
    r3lay mileage       - Update current mileage
    r3lay status        - Show maintenance status
    r3lay history       - Show maintenance history
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .core.maintenance import MaintenanceEntry, MaintenanceLog
from .core.project import ProjectManager

console = Console()


def get_current_mileage(project_path: Path) -> int | None:
    """Get current mileage from project state."""
    pm = ProjectManager(project_path)
    state = pm.load()
    return state.current_mileage if state else None


def prompt_mileage(project_path: Path) -> int:
    """Prompt for mileage, showing current value if available."""
    current = get_current_mileage(project_path)
    if current:
        prompt = f"Mileage [{current}]: "
        response = input(prompt).strip()
        if not response:
            return current
        return int(response)
    else:
        return int(input("Mileage: ").strip())


def log_oil(args: argparse.Namespace) -> int:
    """Log an oil change.

    Args:
        args: Parsed arguments (mileage, products, cost, notes)

    Returns:
        Exit code (0 for success)
    """
    project_path = Path(args.project_path).resolve()
    log = MaintenanceLog(project_path)

    # Get mileage
    if args.mileage:
        mileage = args.mileage
    else:
        mileage = prompt_mileage(project_path)

    # Build entry
    entry = MaintenanceEntry(
        service_type="oil_change",
        mileage=mileage,
        products=args.products.split(",") if args.products else None,
        cost=args.cost,
        notes=args.notes,
        shop=args.shop,
        date=datetime.now(),
    )

    log.add_entry(entry)

    # Update project mileage if this is higher
    pm = ProjectManager(project_path)
    state = pm.load()
    if state and mileage > state.current_mileage:
        pm.update_mileage(mileage)

    console.print(f"[green]✓[/green] Logged oil change at {mileage:,} mi")

    # Show next due
    interval = log.get_interval("oil_change")
    if interval:
        next_due = mileage + interval.interval_miles
        console.print(f"  Next due: {next_due:,} mi")

    return 0


def log_service(args: argparse.Namespace) -> int:
    """Log a scheduled service.

    Args:
        args: Parsed arguments (type, mileage, parts, products, cost, notes)

    Returns:
        Exit code (0 for success)
    """
    project_path = Path(args.project_path).resolve()
    log = MaintenanceLog(project_path)

    # Get service type
    service_type = args.type
    if not service_type:
        # Show available types
        console.print("[bold]Available service types:[/bold]")
        for stype in sorted(log.intervals.keys()):
            interval = log.intervals[stype]
            console.print(f"  {stype}: {interval.description or 'No description'}")
        console.print()
        service_type = input("Service type: ").strip()

    # Get mileage
    if args.mileage:
        mileage = args.mileage
    else:
        mileage = prompt_mileage(project_path)

    # Build entry
    entry = MaintenanceEntry(
        service_type=service_type,
        mileage=mileage,
        parts=args.parts.split(",") if args.parts else None,
        products=args.products.split(",") if args.products else None,
        cost=args.cost,
        notes=args.notes,
        shop=args.shop,
        date=datetime.now(),
    )

    log.add_entry(entry)

    # Update project mileage if this is higher
    pm = ProjectManager(project_path)
    state = pm.load()
    if state and mileage > state.current_mileage:
        pm.update_mileage(mileage)

    console.print(f"[green]✓[/green] Logged {service_type} at {mileage:,} mi")

    # Show next due if interval exists
    interval = log.get_interval(service_type)
    if interval:
        next_due = mileage + interval.interval_miles
        console.print(f"  Next due: {next_due:,} mi")

    return 0


def log_repair(args: argparse.Namespace) -> int:
    """Log a repair.

    Args:
        args: Parsed arguments (description, mileage, parts, cost, notes)

    Returns:
        Exit code (0 for success)
    """
    project_path = Path(args.project_path).resolve()
    log = MaintenanceLog(project_path)

    # Get description
    description = args.description
    if not description:
        description = input("Repair description: ").strip()

    # Get mileage
    if args.mileage:
        mileage = args.mileage
    else:
        mileage = prompt_mileage(project_path)

    # Build entry - repairs use "repair" type with description in notes
    notes = description
    if args.notes:
        notes = f"{description}\n\n{args.notes}"

    entry = MaintenanceEntry(
        service_type="repair",
        mileage=mileage,
        parts=args.parts.split(",") if args.parts else None,
        products=args.products.split(",") if args.products else None,
        cost=args.cost,
        notes=notes,
        shop=args.shop,
        date=datetime.now(),
    )

    log.add_entry(entry)

    # Update project mileage if this is higher
    pm = ProjectManager(project_path)
    state = pm.load()
    if state and mileage > state.current_mileage:
        pm.update_mileage(mileage)

    console.print(f"[green]✓[/green] Logged repair at {mileage:,} mi: {description}")

    return 0


def log_mod(args: argparse.Namespace) -> int:
    """Log a modification.

    Args:
        args: Parsed arguments (description, mileage, parts, cost, notes)

    Returns:
        Exit code (0 for success)
    """
    project_path = Path(args.project_path).resolve()
    log = MaintenanceLog(project_path)

    # Get description
    description = args.description
    if not description:
        description = input("Modification description: ").strip()

    # Get mileage
    if args.mileage:
        mileage = args.mileage
    else:
        mileage = prompt_mileage(project_path)

    # Build entry - mods use "modification" type with description in notes
    notes = description
    if args.notes:
        notes = f"{description}\n\n{args.notes}"

    entry = MaintenanceEntry(
        service_type="modification",
        mileage=mileage,
        parts=args.parts.split(",") if args.parts else None,
        products=args.products.split(",") if args.products else None,
        cost=args.cost,
        notes=notes,
        shop=args.shop,
        date=datetime.now(),
    )

    log.add_entry(entry)

    # Update project mileage if this is higher
    pm = ProjectManager(project_path)
    state = pm.load()
    if state and mileage > state.current_mileage:
        pm.update_mileage(mileage)

    console.print(f"[green]✓[/green] Logged mod at {mileage:,} mi: {description}")

    return 0


def update_mileage(args: argparse.Namespace) -> int:
    """Update current mileage.

    Args:
        args: Parsed arguments (value)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    project_path = Path(args.project_path).resolve()
    pm = ProjectManager(project_path)

    state = pm.load()
    if not state:
        console.print("[red]Error:[/red] No project found. Run r3lay in TUI mode to set up.")
        return 1

    old_mileage = state.current_mileage

    try:
        pm.update_mileage(args.value)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1

    console.print(f"[green]✓[/green] Mileage updated: {old_mileage:,} → {args.value:,} mi")

    # Check for overdue services
    log = MaintenanceLog(project_path)
    overdue = log.get_overdue(args.value)
    if overdue:
        console.print()
        console.print("[yellow]⚠ Overdue services:[/yellow]")
        for sd in overdue[:5]:  # Show max 5
            miles_over = abs(sd.miles_until_due or 0)
            console.print(f"  • {sd.interval.service_type}: {miles_over:,} mi overdue")

    return 0


def show_status(args: argparse.Namespace) -> int:
    """Show maintenance status.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    project_path = Path(args.project_path).resolve()
    pm = ProjectManager(project_path)

    state = pm.load()
    if not state:
        console.print("[yellow]No project found.[/yellow] Run r3lay in TUI mode to set up.")
        return 0

    # Vehicle info
    console.print(f"[bold]{state.profile.display_name}[/bold]")
    console.print(f"Current mileage: {state.current_mileage:,} mi")
    console.print()

    # Maintenance status
    log = MaintenanceLog(project_path)
    upcoming = log.get_upcoming(state.current_mileage, include_never_performed=False)

    if not upcoming:
        console.print("[dim]No maintenance history yet.[/dim]")
        return 0

    # Create table
    table = Table(title="Service Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", justify="right")
    table.add_column("Miles", justify="right")

    for sd in upcoming[:10]:  # Show top 10
        service = sd.interval.service_type.replace("_", " ").title()

        if sd.is_overdue:
            miles_over = abs(sd.miles_until_due or 0)
            status = "[red]OVERDUE[/red]"
            miles = f"-{miles_over:,}"
        else:
            miles_until = sd.miles_until_due or 0
            if miles_until < 1000:
                status = "[yellow]Due Soon[/yellow]"
            else:
                status = "[green]OK[/green]"
            miles = f"+{miles_until:,}"

        table.add_row(service, status, miles)

    console.print(table)

    return 0


def show_history(args: argparse.Namespace) -> int:
    """Show maintenance history.

    Args:
        args: Parsed arguments (limit, type filter)

    Returns:
        Exit code (0 for success)
    """
    project_path = Path(args.project_path).resolve()
    pm = ProjectManager(project_path)

    state = pm.load()
    if not state:
        console.print("[yellow]No project found.[/yellow] Run r3lay in TUI mode to set up.")
        return 0

    log = MaintenanceLog(project_path)
    entries = log.entries

    if not entries:
        console.print("[dim]No maintenance history yet.[/dim]")
        return 0

    # Filter by type if specified
    if args.type:
        filter_type = args.type.lower().replace(" ", "_")
        entries = [e for e in entries if e.service_type.lower() == filter_type]
        if not entries:
            console.print(f"[yellow]No entries found for type '{args.type}'.[/yellow]")
            return 0

    # Sort by date descending (most recent first)
    entries = sorted(entries, key=lambda e: e.date, reverse=True)

    # Apply limit
    limit = args.limit or 20
    entries = entries[:limit]

    # Vehicle info
    console.print(f"[bold]{state.profile.display_name}[/bold]")
    console.print(f"Current mileage: {state.current_mileage:,} mi")
    console.print()

    # Create table
    table = Table(title="Maintenance History")
    table.add_column("Date", style="dim")
    table.add_column("Service", style="cyan")
    table.add_column("Mileage", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Shop/Notes", style="dim")

    for entry in entries:
        date_str = entry.date.strftime("%Y-%m-%d")
        service = entry.service_type.replace("_", " ").title()
        mileage = f"{entry.mileage:,}"
        cost = f"${entry.cost:,.2f}" if entry.cost else "-"

        # Combine shop and notes for last column
        details = []
        if entry.shop:
            details.append(entry.shop)
        if entry.notes:
            # Truncate notes
            note_preview = entry.notes[:30] + "..." if len(entry.notes) > 30 else entry.notes
            note_preview = note_preview.replace("\n", " ")
            details.append(note_preview)

        notes_str = " | ".join(details) if details else "-"

        table.add_row(date_str, service, mileage, cost, notes_str)

    console.print(table)

    # Show totals if we have cost data
    total_cost = sum(e.cost for e in entries if e.cost)
    if total_cost > 0:
        console.print()
        console.print(f"[bold]Total cost:[/bold] ${total_cost:,.2f}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="r3lay",
        description="r³LAY: Garage Terminal - Research Assistant & Maintenance Tracker",
    )
    parser.add_argument(
        "--project",
        "-p",
        dest="project_path",
        default=".",
        help="Path to project directory (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # =========================================================================
    # TUI command (default, no subcommand)
    # =========================================================================

    # =========================================================================
    # log subcommand group
    # =========================================================================
    log_parser = subparsers.add_parser("log", help="Log maintenance entries")
    log_subparsers = log_parser.add_subparsers(dest="log_type", help="Entry type")

    # Common log arguments
    def add_log_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--mileage",
            "-m",
            type=int,
            help="Odometer reading",
        )
        p.add_argument(
            "--cost",
            "-c",
            type=float,
            help="Total cost",
        )
        p.add_argument(
            "--notes",
            "-n",
            help="Additional notes",
        )
        p.add_argument(
            "--shop",
            "-s",
            help="Shop name (or 'DIY')",
        )
        p.add_argument(
            "--parts",
            help="Parts used (comma-separated)",
        )
        p.add_argument(
            "--products",
            help="Products/fluids used (comma-separated)",
        )

    # log oil
    oil_parser = log_subparsers.add_parser("oil", help="Log an oil change")
    add_log_args(oil_parser)
    oil_parser.set_defaults(func=log_oil)

    # log service
    service_parser = log_subparsers.add_parser("service", help="Log a scheduled service")
    service_parser.add_argument(
        "type",
        nargs="?",
        help="Service type (e.g., tire_rotation, brake_pads)",
    )
    add_log_args(service_parser)
    service_parser.set_defaults(func=log_service)

    # log repair
    repair_parser = log_subparsers.add_parser("repair", help="Log a repair")
    repair_parser.add_argument(
        "description",
        nargs="?",
        help="Repair description",
    )
    add_log_args(repair_parser)
    repair_parser.set_defaults(func=log_repair)

    # log mod
    mod_parser = log_subparsers.add_parser("mod", help="Log a modification")
    mod_parser.add_argument(
        "description",
        nargs="?",
        help="Modification description",
    )
    add_log_args(mod_parser)
    mod_parser.set_defaults(func=log_mod)

    # =========================================================================
    # mileage command
    # =========================================================================
    mileage_parser = subparsers.add_parser("mileage", help="Update current mileage")
    mileage_parser.add_argument(
        "value",
        type=int,
        help="New odometer reading",
    )
    mileage_parser.set_defaults(func=update_mileage)

    # =========================================================================
    # status command
    # =========================================================================
    status_parser = subparsers.add_parser("status", help="Show maintenance status")
    status_parser.set_defaults(func=show_status)

    # =========================================================================
    # history command
    # =========================================================================
    history_parser = subparsers.add_parser("history", help="Show maintenance history")
    history_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Maximum entries to show (default: 20)",
    )
    history_parser.add_argument(
        "--type",
        "-t",
        help="Filter by service type (e.g., 'oil_change', 'repair')",
    )
    history_parser.set_defaults(func=show_history)

    return parser


def run_cli(args: list[str] | None = None) -> int | None:
    """Run the CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Handle log subcommand without type
    if parsed.command == "log" and not parsed.log_type:
        # Show log help
        parser.parse_args(["log", "--help"])
        return 0

    # Handle subcommand functions
    if hasattr(parsed, "func"):
        try:
            return parsed.func(parsed)
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]")
            return 130
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1

    # No subcommand = show help or launch TUI
    return None  # Signal to launch TUI


__all__ = [
    "create_parser",
    "run_cli",
    "log_oil",
    "log_service",
    "log_repair",
    "log_mod",
    "update_mileage",
    "show_status",
    "show_history",
]
