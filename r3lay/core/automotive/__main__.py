"""CLI interface for testing automotive module.

Usage:
    python -m r3lay.core.automotive P0420
    python -m r3lay.core.automotive --search catalyst
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from r3lay.core.automotive.obd2 import OBD2Database

console = Console()


def print_code_details(dtc):
    """Print detailed diagnostic code information."""
    # Determine border style based on severity
    severity_styles = {
        "critical": "red",
        "moderate": "yellow",
        "minor": "green",
        "info": "blue",
    }
    border_style = severity_styles.get(dtc.severity.value, "white")

    # Display header panel
    console.print(
        Panel(
            f"[bold]{dtc.code}[/bold]: {dtc.description}",
            title=f"[{dtc.severity.value.upper()}] {dtc.protocol}",
            border_style=border_style,
        )
    )

    # Common causes
    console.print("\n[bold]Common Causes:[/bold]")
    for i, cause in enumerate(dtc.common_causes, 1):
        console.print(f"  {i}. {cause}")

    # Diagnostic steps
    console.print("\n[bold]Diagnostic Steps:[/bold]")
    for step in dtc.diagnostic_steps:
        console.print(f"  • {step}")

    # Related codes
    if dtc.related_codes:
        console.print(f"\n[bold]Related Codes:[/bold] {', '.join(dtc.related_codes)}")

    # Forum links
    if dtc.forum_links:
        console.print("\n[bold]Community Discussions:[/bold]")
        for link in dtc.forum_links:
            console.print(f"  • {link}")

    # Notes
    if dtc.notes:
        console.print(f"\n[dim]{dtc.notes}[/dim]")


def print_search_results(results):
    """Print search results in a table."""
    if not results:
        console.print("[yellow]No codes found matching your search.[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("Code", style="cyan", no_wrap=True)
    table.add_column("Severity", style="magenta")
    table.add_column("Description")

    for dtc in results[:10]:  # Limit to 10 results
        severity_style = {
            "critical": "[red]●[/red]",
            "moderate": "[yellow]●[/yellow]",
            "minor": "[green]●[/green]",
            "info": "[blue]●[/blue]",
        }.get(dtc.severity.value, "○")

        table.add_row(dtc.code, severity_style, dtc.description)

    console.print(table)

    if len(results) > 10:
        console.print(f"\n[dim]Showing 10 of {len(results)} results[/dim]")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        console.print("[yellow]Usage:[/yellow]")
        console.print("  python -m r3lay.core.automotive <code>")
        console.print("  python -m r3lay.core.automotive --search <query>")
        console.print("\n[yellow]Examples:[/yellow]")
        console.print("  python -m r3lay.core.automotive P0420")
        console.print("  python -m r3lay.core.automotive --search catalyst")
        sys.exit(1)

    # Load database
    knowledge_path = Path.home() / ".r3lay" / "knowledge"
    db = OBD2Database(knowledge_path)

    if len(db) == 0:
        console.print("[red]No OBD2 database found![/red]")
        console.print(f"[yellow]Expected location:[/yellow] {db.knowledge_path}")
        console.print(
            "\n[dim]Create the directory and add JSON database files.[/dim]"
        )
        sys.exit(1)

    console.print(f"[dim]Loaded {len(db)} codes from {db.knowledge_path}[/dim]\n")

    # Handle search mode
    if sys.argv[1] == "--search":
        if len(sys.argv) < 3:
            console.print("[red]Error: --search requires a query[/red]")
            sys.exit(1)

        query = " ".join(sys.argv[2:])
        results = db.search(query)
        print_search_results(results)
        sys.exit(0)

    # Handle code lookup
    code = sys.argv[1].upper()
    dtc = db.lookup(code)

    if not dtc:
        console.print(f"[red]Code {code} not found in database[/red]")
        console.print("\n[yellow]Try searching:[/yellow]")
        console.print(f"  python -m r3lay.core.automotive --search {code}")
        sys.exit(1)

    print_code_details(dtc)

    # Show related codes if any
    related = db.get_related(code)
    if related:
        console.print(f"\n[dim]Use --search to explore related codes: {', '.join([r.code for r in related])}[/dim]")


if __name__ == "__main__":
    main()
