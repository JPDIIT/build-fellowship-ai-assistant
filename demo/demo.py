#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from openai import OpenAI
from anthropic import Anthropic

from analytics_assistant.agent import DataAnalyticsAgent
from analytics_assistant.cli import DataAnalyticsCLI


console = Console()


def setup_llm_client(provider: str = "openai") -> tuple:
    """
    Set up LLM client with API key validation.

    Returns:
        (provider_name, client)
    """
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            console.print("Set it with: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        return "OpenAI GPT-4", OpenAI(api_key=api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[red]Error: ANTHROPIC_API_KEY environment variable not set[/red]")
            console.print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            sys.exit(1)
        return "Anthropic Claude", Anthropic(api_key=api_key)

    else:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        sys.exit(1)


def check_data_files() -> bool:
    """Check if demo data files exist."""
    data_dir = Path(__file__).parent / "data"
    sales_file = data_dir / "food.csv"
    customers_file = data_dir / "food.csv"

    if not sales_file.exists() or not customers_file.exists():
        console.print("[red]Error: Demo data files not found![/red]")
        console.print(f"Expected files:")
        console.print(f"  - {sales_file}")
        console.print(f"  - {customers_file}")
        console.print("\nGenerate them with: python generate_demo_data.py")
        return False

    return True


def print_demo_header(title: str, description: str):
    """Print a formatted demo section header."""
    console.print("\n")
    console.print(Panel(
        Markdown(f"# {title}\n\n{description}"),
        border_style="bold cyan"
    ))


def run_query_with_feedback(agent: DataAnalyticsAgent, query: str, show_code: bool = False):
    """
    Run a query and display results with nice formatting.

    Args:
        agent: DataAnalyticsAgent instance
        query: Natural language query
        show_code: Whether to display generated code
    """
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")

    with console.status("[bold green]Processing...", spinner="dots"):
        result = agent.run(query)

    if result["status"] == "success":
        # Show answer
        console.print(Panel(
            Markdown(result["answer"]),
            title="[green]Answer[/green]",
            border_style="green"
        ))

        # Show code if requested
        if show_code and result.get("conversation_history"):
            for turn in result["conversation_history"]:
                if "generated_code" in turn.get("result", {}):
                    from rich.syntax import Syntax
                    code = turn["result"]["generated_code"]
                    console.print("\n[bold]Generated Code:[/bold]")
                    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

        # Show iterations
        console.print(f"[dim]Completed in {result.get('iterations', 0)} iterations[/dim]")

        return True
    else:
        console.print(Panel(
            f"[red]{result['error_message']}[/red]",
            title="[red]Error[/red]",
            border_style="red"
        ))
        return False


def run_quick_demo():
    """Run a quick 5-minute demo."""
    console.print(Panel(
        Markdown("""
        # Quick Demo (5 minutes)

        This demo showcases:
        1. Loading datasets
        2. Quick data exploration
        3. Simple analysis
        4. Visualization
        """),
        title="[bold green]Quick Demo[/bold green]",
        border_style="green"
    ))

    # Setup
    provider_name, llm_client = setup_llm_client()
    console.print(f"[green]Using {provider_name}[/green]\n")

    if not check_data_files():
        return

    agent = DataAnalyticsAgent(llm_client=llm_client, verbose=False)

    data_dir = Path(__file__).parent / "data"

    # Scenario 1: Load data
    print_demo_header(
        "Step 1: Loading Data",
        "Load the e-commerce sales data"
    )
    run_query_with_feedback(
        agent,
        f"Load the CSV file at '{data_dir / 'food.csv'}' with alias 'sales'"
    )

    time.sleep(1)

    # Scenario 2: Quick exploration
    print_demo_header(
        "Step 2: Data Exploration",
        "Understand what's in the dataset"
    )
    run_query_with_feedback(
        agent,
        "What columns are in the sales data? Show me a few sample rows."
    )

    time.sleep(1)

    # Scenario 3: Simple analysis
    print_demo_header(
        "Step 3: Analysis",
        "Find the top products by revenue"
    )
    run_query_with_feedback(
        agent,
        "What are the top 5 products by total revenue?",
        show_code=True
    )

    time.sleep(1)

    # Scenario 4: Visualization
    print_demo_header(
        "Step 4: Visualization",
        "Create a visualization"
    )
    result = agent.run("Show revenue by category as a bar chart")
    if result["status"] == "success":
        console.print("[green]Visualization created successfully![/green]")
        if result.get("visualizations"):
            console.print(f"[green]Generated {len(result['visualizations'])} chart(s)[/green]")

    console.print("\n[bold green]Quick demo complete![/bold green]")


def run_full_demo():
    """Run a comprehensive 15-minute demo."""
    console.print(Panel(
        Markdown("""
        # Full Demo (15 minutes)

        This demo showcases:
        1. Dataset management (load, inspect)
        2. Complex analytics with code generation
        3. Multi-table analysis (joins)
        4. Conversational context
        5. Multiple visualizations
        """),
        title="[bold green]Full Demo[/bold green]",
        border_style="green"
    ))

    # Setup
    provider_name, llm_client = setup_llm_client()
    console.print(f"[green]Using {provider_name}[/green]\n")

    if not check_data_files():
        return

    agent = DataAnalyticsAgent(llm_client=llm_client, verbose=False)

    data_dir = Path(__file__).parent / "data"

    # Part 1: Load both datasets
    print_demo_header(
        "Part 1: Loading Multiple Datasets",
        "Load sales and customer data"
    )

    run_query_with_feedback(
        agent,
        f"Load '{data_dir / 'food.csv'}' as 'sales' and '{data_dir / 'food.csv'}' as 'customers'"
    )

    time.sleep(1)

    run_query_with_feedback(agent, "List all loaded datasets")

    time.sleep(1)

    # Part 2: Data exploration
    print_demo_header(
        "Part 2: Dataset Inspection",
        "Examine the structure of our data"
    )

    run_query_with_feedback(
        agent,
        "Inspect the sales dataset and show me statistics for numeric columns"
    )

    time.sleep(1)

    # Part 3: Simple analysis
    print_demo_header(
        "Part 3: Basic Analytics",
        "Aggregate and filter data"
    )

    run_query_with_feedback(
        agent,
        "What are the top 5 products by revenue?",
        show_code=True
    )

    time.sleep(1)

    run_query_with_feedback(
        agent,
        "How much revenue did we make in each category?"
    )

    time.sleep(1)

    # Part 4: Multi-table analysis
    print_demo_header(
        "Part 4: Multi-Table Analysis",
        "Join sales and customer data"
    )

    run_query_with_feedback(
        agent,
        "Join sales and customers data. Show me total revenue by customer segment (Premium vs Standard).",
        show_code=True
    )

    time.sleep(1)

    # Part 5: Conversational context
    print_demo_header(
        "Part 5: Conversational Context",
        "The agent remembers previous queries"
    )

    run_query_with_feedback(
        agent,
        "What's the average order value for each segment?"
    )

    time.sleep(1)

    # Part 6: Time series
    print_demo_header(
        "Part 6: Time Series Analysis",
        "Analyze trends over time"
    )

    run_query_with_feedback(
        agent,
        "Show me monthly revenue trend for 2024",
        show_code=True
    )

    time.sleep(1)

    # Part 7: Visualizations
    print_demo_header(
        "Part 7: Visualizations",
        "Create multiple charts"
    )

    run_query_with_feedback(
        agent,
        "Create a bar chart showing revenue by category"
    )

    time.sleep(1)

    run_query_with_feedback(
        agent,
        "Show the monthly revenue trend as a line chart"
    )

    time.sleep(1)

    # Part 8: Complex query
    print_demo_header(
        "Part 8: Complex Analysis",
        "Multi-step reasoning"
    )

    run_query_with_feedback(
        agent,
        "Which product categories are most popular among Premium customers? Show the top 3.",
        show_code=True
    )

    console.print("\n[bold green]Full demo complete![/bold green]")
    console.print(f"[dim]The agent completed all tasks using the ReAct pattern[/dim]")


def run_interactive(provider: str = "openai"):
    """Run interactive CLI mode."""

    cli = DataAnalyticsCLI(llm_provider=provider)
    cli.run()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Workshop 07 Data Analytics Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    # Interactive mode
  python demo.py --scenario quick   # Quick 5-minute demo
  python demo.py --scenario full    # Full 15-minute demo
  python demo.py --provider anthropic --scenario quick
        """
    )

    parser.add_argument(
        "--scenario",
        choices=["quick", "full", "interactive"],
        help="Demo scenario to run (default: interactive)"
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)"
    )

    args = parser.parse_args()

    # Set default provider for demo scenarios
    if args.scenario in ["quick", "full"]:
        os.environ.setdefault("LLM_PROVIDER", args.provider)

    try:
        if args.scenario == "quick":
            run_quick_demo()
        elif args.scenario == "full":
            run_full_demo()
        else:
            # Default to interactive
            run_interactive(args.provider)

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
