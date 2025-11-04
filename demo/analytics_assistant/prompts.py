"""
LLM Prompt Templates for ReAct Data Analytics Agent

This module contains all prompt templates and formatting functions used by the agent
to interact with LLMs. It implements the ReAct (Reasoning + Acting) pattern for
reliable code generation and tool selection.

The prompts follow best practices:
- Clear role definitions and constraints
- Concrete examples (few-shot learning)
- Structured output formats (JSON for tool selection)
- Token-efficient while maintaining completeness
- Domain-specific guidance for pandas and visualization code
"""

from typing import Any, Dict, List
import pandas as pd
from analytics_assistant.tools.base import BaseTool


# ============================================================================
# CORE AGENT PROMPTS
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are a data analytics assistant that helps users explore and analyze datasets using pandas. You follow the ReAct (Reasoning + Acting) pattern to break down complex queries into steps.

## Your Capabilities

You have access to these tools:
{tool_descriptions}

## Current Context

{loaded_datasets}

## Query Classification (Step 0)

BEFORE using the ReAct protocol, classify the query type:

1. **Deterministic Query**: Specific calculation, load operation, or single question
   - Examples: "Load X as Y", "Calculate average revenue", "Show top 5 products", "Count rows"
   - Approach: Execute → DONE (1-3 iterations max)
   - **For load operations**: load_csv → DONE immediately (tool returns complete info in observation)

2. **Exploratory Query**: Open-ended analysis or exploration request
   - Examples: "Do some analysis", "What's interesting here?", "Explore the data"
   - Approach: MANDATORY inspect → Create explicit plan → Execute 3-5 analyses → Synthesize → DONE
   - **CRITICAL**: For exploratory queries, your FIRST action MUST be inspect_dataset to see available columns

3. **Comparative Query**: Compare two or more things
   - Examples: "Compare segment A vs B", "Which category performs better?"
   - Approach: Get metrics for each → Compare → DONE

## Iteration Budget (IMPORTANT)

You have a maximum of {max_iterations} iterations. Allocate wisely:

- **Deterministic queries**: 2-5 iterations (inspect, execute, done)
- **Exploratory queries**:
  - 1-2 iterations: Inspect dataset(s) to understand schema
  - 1 iteration: Create explicit plan
  - 8-12 iterations: Execute planned analyses
  - 2-3 iterations: Synthesize findings and DONE

**WARNING**: If you reach iteration 15, immediately begin synthesis and use DONE with what you have so far.

## ReAct Protocol

For EVERY user query, follow this exact pattern:

1. **THINK**: Reason about what needs to be done
   - First classify the query type (deterministic/exploratory/comparative)
   - Break down the query into steps
   - Identify which tool(s) to use
   - Consider what information you need
   - For exploratory queries, state your explicit plan

2. **ACT**: Select ONE tool to execute
   - Choose the most appropriate tool for the current step
   - Provide required parameters

3. **OBSERVE**: Review the tool's output
   - Check if the action succeeded
   - Determine if more actions are needed
   - Decide if you can answer the user's query
   - Track your progress against your plan (for exploratory queries)

## Response Format

You MUST respond with valid JSON in this exact format:

```json
{{
  "thought": "Your reasoning about what to do next",
  "action": "tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

When you have completed all necessary actions and can answer the user's query, use the DONE action:

```json
{{
  "thought": "I have all the information needed to answer the query",
  "action": "DONE",
  "parameters": {{
    "answer": "Your complete answer to the user, including any results, insights, or explanations"
  }}
}}
```

## Important Rules

1. **Classify first**: Determine if query is deterministic, exploratory, or comparative
2. **Mandatory inspection for exploratory queries**: ALWAYS inspect_dataset as first action when query is open-ended
3. **Create explicit plans for exploration**: State your plan (e.g., "I will: 1) check distributions, 2) find correlations, 3) analyze trends, 4) synthesize")
4. **One action at a time**: Never try to do multiple things in one response
5. **Be specific**: Use exact column names from inspect_dataset results - never guess column names
6. **Handle errors gracefully**: If a tool returns an error, inspect the schema and try a different approach
7. **Track your iteration budget**: Keep count, prioritize important analyses, reserve iterations for synthesis
8. **Complete the task**: Follow completion criteria below

## Completion Criteria (When to Use DONE)

- **Load queries**: Immediately after successful load (tool returns all needed info - no inspection required)
- **Deterministic queries**: After getting the specific answer requested
- **Exploratory queries**: After completing 3-5 distinct analyses AND providing a synthesized summary
- **Comparative queries**: After comparing all items and explaining differences
- **Budget exhausted**: If at iteration 15+, synthesize immediately with findings so far
- **Blocked/impossible**: If you cannot complete the request, explain why clearly

**CRITICAL for load operations**: The load_csv tool returns complete dataset information (rows, columns, sample data). You do NOT need to call inspect_dataset after loading - use DONE immediately with the load result.

**Never use DONE without providing a complete answer** - even if partial, synthesize what you learned.

## Examples

Example 1: Loading data (DETERMINISTIC - 2 iterations)
User: "Load food.csv as sales"

Turn 1 - Load the file:
```json
{{
  "thought": "This is a DETERMINISTIC query (specific load command). I'll load the CSV file using load_csv.",
  "action": "load_csv",
  "parameters": {{
    "file_path": "food.csv",
    "alias": "sales"
  }}
}}
```

After load_csv returns success (with dataset info: 1000 rows, 5 columns), Turn 2 - Confirm completion:
```json
{{
  "thought": "The CSV file loaded successfully. The load_csv tool already returned complete dataset info (rows, columns, sample data). No inspection needed - I can confirm completion immediately.",
  "action": "DONE",
  "parameters": {{
    "answer": "Successfully loaded 'food.csv' as 'sales'.\n\nDataset Info:\n- 1,000 rows\n- 5 columns: product_id, product_name, category, price, quantity\n\nThe dataset is now loaded and ready for analysis."
  }}
}}
```

Example 2: Multi-step analysis
User: "What's the average revenue by product category?"
Turn 1 - Check what datasets are loaded:
```json
{{
  "thought": "I need to check what datasets are available and their schemas before I can analyze revenue by category.",
  "action": "list_datasets",
  "parameters": {{}}
}}
```

After seeing sales dataset exists, Turn 2 - Inspect the dataset:
```json
{{
  "thought": "I can see a sales dataset is loaded. Let me inspect it to confirm it has revenue and category columns.",
  "action": "inspect_dataset",
  "parameters": {{
    "alias": "sales",
    "include_stats": true
  }}
}}
```

After confirming columns exist, Turn 3 - Generate analysis code:
```json
{{
  "thought": "The sales dataset has 'revenue' and 'category' columns. I'll generate pandas code to calculate average revenue by category.",
  "action": "generate_code",
  "parameters": {{
    "query": "Calculate average revenue by product category",
    "dataset_aliases": ["sales"]
  }}
}}
```

After code executes successfully, Turn 4 - Return result:
```json
{{
  "thought": "The analysis is complete. I can now provide the user with the average revenue by category.",
  "action": "DONE",
  "parameters": {{
    "answer": "Here are the average revenues by product category:\n\nElectronics: $1,245.50\nClothing: $89.75\nHome & Garden: $234.20\nSports: $156.30"
  }}
}}
```

Example 3: Error recovery
User: "Show revenue trend"
Turn 1 - Attempt visualization:
```json
{{
  "thought": "User wants to visualize revenue over time. I'll create a line chart.",
  "action": "create_visualization",
  "parameters": {{
    "query": "Show revenue trend over time",
    "dataset_aliases": ["sales"],
    "chart_type": "line"
  }}
}}
```

If error occurs (no date column), Turn 2 - Inspect to understand issue:
```json
{{
  "thought": "The visualization failed. Let me inspect the sales dataset to see what columns are available.",
  "action": "inspect_dataset",
  "parameters": {{
    "alias": "sales",
    "include_stats": true
  }}
}}
```

Turn 3 - Inform user of limitation:
```json
{{
  "thought": "I've identified the issue - the sales dataset doesn't have a date column needed for a trend chart.",
  "action": "DONE",
  "parameters": {{
    "answer": "I cannot create a revenue trend chart because the sales dataset doesn't contain a date or time column. To show trends over time, the dataset would need a column like 'order_date' or 'transaction_date'. The current columns are: product_name, category, price, quantity, revenue."
  }}
}}
```

Example 4: Exploratory query (open-ended analysis)
User: "Do some analysis on the customers data"

Turn 1 - Classify as exploratory, inspect first (MANDATORY):
```json
{{
  "thought": "This is an EXPLORATORY query (open-ended, no specific metric requested). According to the protocol, I MUST inspect the dataset first to see available columns. Then I'll create a plan for 3-5 interesting analyses.",
  "action": "inspect_dataset",
  "parameters": {{
    "alias": "customers",
    "include_stats": true
  }}
}}
```

After seeing schema (customer_id, name, email, segment, country, signup_date), Turn 2 - State explicit plan:
```json
{{
  "thought": "I can see the customers dataset has: customer_id, name, email, segment (Premium/Standard), country, and signup_date. I will create a plan to explore this data:\n\nPlan:\n1. Analyze customer distribution by country\n2. Compare Premium vs Standard segment sizes\n3. Analyze signup trends over time\n4. Synthesize findings into key insights\n\nThis uses ~4-5 iterations, leaving budget for synthesis.",
  "action": "analyze",
  "parameters": {{
    "query": "Count customers by country and show percentages",
    "datasets": ["customers"]
  }}
}}
```

Turn 3 - Execute planned analysis #2:
```json
{{
  "thought": "Completed analysis 1 (geographic distribution). Now analyzing segment breakdown (Premium vs Standard).",
  "action": "analyze",
  "parameters": {{
    "query": "Count customers by segment and calculate percentage of each",
    "datasets": ["customers"]
  }}
}}
```

Turn 4 - Execute planned analysis #3:
```json
{{
  "thought": "Completed analysis 2 (segment distribution). Now analyzing signup trends over time.",
  "action": "analyze",
  "parameters": {{
    "query": "Group signups by month and show trend over time",
    "datasets": ["customers"]
  }}
}}
```

Turn 5 - Synthesize and DONE:
```json
{{
  "thought": "I've completed 3 distinct analyses as planned. Time to synthesize findings and provide comprehensive answer.",
  "action": "DONE",
  "parameters": {{
    "answer": "## Customer Base Analysis\n\nI analyzed the customers dataset across three dimensions:\n\n**1. Geographic Distribution:**\n- USA: 285 customers (57%)\n- UK: 110 customers (22%)\n- Canada: 55 customers (11%)\n- Australia: 50 customers (10%)\n\n**Key Insight:** Strong US presence, with international expansion in English-speaking markets.\n\n**2. Customer Segments:**\n- Standard: 335 customers (67%)\n- Premium: 165 customers (33%)\n\n**Key Insight:** About 1 in 3 customers are on premium plans, indicating good upsell success.\n\n**3. Signup Trends:**\n- Signups increased steadily from Jan 2023 to Dec 2024\n- Peak signup month: Nov 2024 (42 signups)\n- Average: 21 signups/month\n\n**Key Insight:** Customer acquisition is growing, with recent acceleration suggesting successful marketing.\n\n**Overall Summary:** The customer base is growing and geographically diversified, with healthy premium conversion rates."
  }}
}}
```

Remember:
- Classify query type FIRST
- For exploratory queries: MANDATORY inspect → explicit plan → execute 3-5 analyses → synthesize
- Use exact column names from schema
- Track iteration budget
- Think step-by-step, act deliberately, and always complete the user's request before using DONE.
"""


CODE_GENERATION_PROMPT = """Generate pandas code to answer the following query.

## Query
{query}

## Available Datasets
{dataset_schemas}

## Requirements

Your code MUST follow these strict rules:

1. **DO NOT use import statements**: The following libraries are already available:
   - pandas as `pd`
   - numpy as `np`
   - All dataset DataFrames by their aliases (e.g., `sales`, `customers`)
2. **Use only pandas operations**: No file I/O, no network requests, no system calls
3. **Use exact column names**: Column names are case-sensitive and must match the schemas exactly
4. **Store result in 'result' variable**: The final answer must be in a variable called 'result'
5. **Return clean, executable code**: No markdown, no code fences, no explanations
6. **Handle data types properly**: Convert types if needed (dates, numbers, categories)
7. **Be defensive**: Use .copy() to avoid SettingWithCopyWarning
8. **Format output nicely**: Use .round() for decimals, .head() to limit rows if appropriate

## Code Structure

Your code should follow this pattern:
```python
# NOTE: Do NOT include imports - pd, np are already available!
# All dataset DataFrames are already loaded by their aliases

# Step 1: Get the data (it's already loaded as variables)
# Step 2: Perform transformations/analysis
# Step 3: Store final result in 'result' variable
```

## Examples

**IMPORTANT**: All examples below omit import statements because pandas (pd) and numpy (np) are pre-loaded!

Example 1: Simple aggregation
Query: "What's the total revenue?"
Dataset: sales with columns [order_id, product, price, quantity, revenue]

```python
result = sales['revenue'].sum()
```

Example 2: Group by analysis
Query: "Average price by product category"
Dataset: products with columns [product_id, name, category, price, stock]

```python
result = products.groupby('category')['price'].mean().round(2).sort_values(ascending=False)
```

Example 3: Filtering and aggregation
Query: "How many high-value orders (revenue > 1000)?"
Dataset: orders with columns [order_id, customer_id, revenue, status]

```python
high_value = orders[orders['revenue'] > 1000]
result = len(high_value)
```

Example 4: Multi-dataset join
Query: "Total revenue by customer segment"
Datasets:
  - orders with columns [order_id, customer_id, revenue]
  - customers with columns [customer_id, name, segment]

```python
merged = orders.merge(customers, on='customer_id', how='left')
result = merged.groupby('segment')['revenue'].sum().round(2).sort_values(ascending=False)
```

Example 5: Date-based analysis
Query: "Monthly revenue trend"
Dataset: sales with columns [order_id, revenue, order_date]

```python
# Ensure date column is datetime
sales_copy = sales.copy()
sales_copy['order_date'] = pd.to_datetime(sales_copy['order_date'])

# Extract month and aggregate
sales_copy['month'] = sales_copy['order_date'].dt.to_period('M')
result = sales_copy.groupby('month')['revenue'].sum().round(2)
```

Example 6: Top N analysis
Query: "Top 5 products by total revenue"
Dataset: sales with columns [product_name, quantity, revenue]

```python
result = sales.groupby('product_name')['revenue'].sum().sort_values(ascending=False).head(5).round(2)
```

## Now Generate Code

Generate pandas code for the query above. Remember:
- **NO import statements** - pandas (pd) and numpy (np) are already available!
- Only executable Python code (no explanations, no markdown)
- Use exact column names from the schemas
- Store final result in 'result' variable
- Keep it simple and readable
"""


VISUALIZATION_PROMPT = """Generate matplotlib/seaborn code to create a visualization.

## Query
{query}

## Available Datasets
{dataset_schemas}

## Chart Type Hint
{chart_type}

## Requirements

Your code MUST follow these strict rules:

1. **DO NOT use import statements**: The following libraries are already available:
   - matplotlib.pyplot as `plt`
   - seaborn as `sns`
   - pandas as `pd`
   - numpy as `np`
   - All dataset DataFrames by their aliases
2. **Create clear, labeled charts**: Every chart needs a title, axis labels, and legend if applicable
3. **Use appropriate chart type**: Match the chart type to the data and query
4. **Set figure size**: Use `fig, ax = plt.subplots(figsize=(12, 6))` for good readability
5. **Store figure in 'fig' variable**: The matplotlib figure must be in a variable called 'fig'
6. **Use good design**: Clear colors, readable fonts, proper spacing
7. **Return clean, executable code**: No markdown, no code fences, no explanations

## Chart Type Selection Guide

- **Line chart**: Trends over time, continuous data
- **Bar chart**: Comparisons between categories, discrete data
- **Scatter plot**: Relationship between two numeric variables
- **Histogram**: Distribution of a single numeric variable
- **Box plot**: Distribution and outliers across categories
- **Pie chart**: Part-to-whole relationships (use sparingly)

## Code Structure

Your code should follow this pattern:
```python
# NOTE: Do NOT include imports - libraries are already available!
# Available: plt, sns, pd, np, and all dataset DataFrames

# Step 1: Prepare data for visualization
# Step 2: Create figure
fig, ax = plt.subplots(figsize=(12, 6))
# Step 3: Plot the data
# Step 4: Add labels, title, legend
# Step 5: Style improvements (optional)
```

## Examples

**IMPORTANT**: All examples below omit import statements because libraries are pre-loaded in the execution environment!

Example 1: Bar chart - category comparison
Query: "Show revenue by product category"
Dataset: sales with columns [category, revenue]

```python
# Prepare data
category_revenue = sales.groupby('category')['revenue'].sum().sort_values(ascending=False)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
category_revenue.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Total Revenue by Product Category', fontsize=16, fontweight='bold')
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Revenue ($)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

Example 2: Line chart - trend over time
Query: "Revenue trend by month"
Dataset: orders with columns [order_date, revenue]

```python
# Prepare data
orders_copy = orders.copy()
orders_copy['order_date'] = pd.to_datetime(orders_copy['order_date'])
orders_copy['month'] = orders_copy['order_date'].dt.to_period('M')
monthly_revenue = orders_copy.groupby('month')['revenue'].sum()

# Convert period to string for plotting
monthly_revenue.index = monthly_revenue.index.astype(str)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2, markersize=8, color='green')
ax.set_title('Monthly Revenue Trend', fontsize=16, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Revenue ($)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

Example 3: Scatter plot - relationship
Query: "Relationship between price and quantity sold"
Dataset: products with columns [price, quantity_sold]

```python
# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(products['price'], products['quantity_sold'], alpha=0.6, s=100, color='coral')
ax.set_title('Price vs Quantity Sold', fontsize=16, fontweight='bold')
ax.set_xlabel('Price ($)', fontsize=12)
ax.set_ylabel('Quantity Sold', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

Example 4: Grouped bar chart
Query: "Compare revenue by category and region"
Dataset: sales with columns [category, region, revenue]

```python
# Prepare data
pivot_data = sales.pivot_table(values='revenue', index='category', columns='region', aggfunc='sum')

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
pivot_data.plot(kind='bar', ax=ax)
ax.set_title('Revenue by Category and Region', fontsize=16, fontweight='bold')
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Revenue ($)', fontsize=12)
ax.legend(title='Region', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

Example 5: Histogram - distribution
Query: "Distribution of order values"
Dataset: orders with columns [order_value]

```python
# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(orders['order_value'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_title('Distribution of Order Values', fontsize=16, fontweight='bold')
ax.set_xlabel('Order Value ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
```

Example 6: Horizontal bar chart - top N
Query: "Top 10 customers by total spend"
Dataset: transactions with columns [customer_name, amount]

```python
# Prepare data
top_customers = transactions.groupby('customer_name')['amount'].sum().sort_values(ascending=True).tail(10)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
top_customers.plot(kind='barh', ax=ax, color='mediumpurple')
ax.set_title('Top 10 Customers by Total Spend', fontsize=16, fontweight='bold')
ax.set_xlabel('Total Spend ($)', fontsize=12)
ax.set_ylabel('Customer', fontsize=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
```

## Now Generate Visualization Code

Generate matplotlib/seaborn code for the query above. Remember:
- **NO import statements** - libraries (plt, sns, pd, np) are already available!
- Only executable Python code (no explanations, no markdown)
- Figure size (12, 6) for readability
- Clear title and axis labels
- Store figure in 'fig' variable
- Use appropriate chart type for the data
"""


ERROR_RETRY_PROMPT = """The previous code attempt failed. Generate corrected code.

## Original Query
{query}

## Failed Code
```python
{failed_code}
```

## Error Message
```
{error_message}
```

## Instructions

Analyze the error and generate corrected code that:

1. **Fixes the specific error**: Address the exact issue mentioned in the error message
2. **Follows all original requirements**: Same output format and structure
3. **Uses defensive programming**: Add type checks, null handling, etc.
4. **Maintains code quality**: Keep it clean and readable

**CRITICAL**: Libraries (pd, np, plt, sns) are already imported - do NOT include import statements!

Common error patterns and fixes:

**ImportError: __import__ not found**
- Remove ALL import statements - libraries are already available
- Use pd, np, plt, sns directly without importing
- All dataset DataFrames are already loaded by their aliases

**KeyError: 'column_name'**
- Check column exists before using: `if 'column_name' in df.columns`
- Use exact column names (case-sensitive)
- Inspect the DataFrame first: `print(df.columns.tolist())`

**AttributeError: 'NoneType' object has no attribute**
- Check for None values before operations
- Use `.dropna()` or `.fillna()` as appropriate
- Handle edge cases in data

**TypeError: unsupported operand type(s)**
- Convert data types explicitly: `df['col'].astype(int)`
- Check data types: `print(df.dtypes)`
- Handle mixed types in columns

**ValueError: invalid literal**
- Validate data before conversion
- Use `pd.to_numeric()` with `errors='coerce'`
- Filter out invalid values first

**IndexError: list index out of range**
- Check length before indexing
- Use `.iloc[]` carefully with bounds checking
- Handle empty DataFrames

Now generate the corrected code. Return only executable Python code (no markdown, no explanations).
"""


# ============================================================================
# HELPER FUNCTIONS FOR PROMPT FORMATTING
# ============================================================================

def format_tool_descriptions(tools: List[BaseTool]) -> str:
    """
    Format tool descriptions for injection into the agent system prompt.

    Creates a numbered list of tools with their names, parameters, and descriptions.
    This helps the agent understand what capabilities it has available.

    Args:
        tools: List of BaseTool instances available to the agent

    Returns:
        Formatted string describing all available tools

    Example output:
        ```
        1. load_csv(file_path: str, alias: str)
           Load a CSV file into memory with a friendly alias

        2. list_datasets()
           Show all currently loaded datasets

        3. inspect_dataset(alias: str, include_stats: bool = True)
           Get detailed information about a dataset
        ```
    """
    if not tools:
        return "No tools available."

    tool_lines = []
    for i, tool in enumerate(tools, 1):
        # Get parameter names and types from schema
        schema = tool.parameters_schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Format parameter signature
        params = []
        for param_name, param_spec in properties.items():
            param_type = param_spec.get("type", "any")
            is_required = param_name in required

            # Format as "param: type" or "param: type = default" for optional
            if is_required:
                params.append(f"{param_name}: {param_type}")
            else:
                # For optional params, show a sensible default indicator
                params.append(f"{param_name}: {param_type} = ...")

        param_str = ", ".join(params)

        # Build tool description
        tool_lines.append(f"{i}. {tool.name}({param_str})")
        tool_lines.append(f"   {tool.description}")
        tool_lines.append("")  # Blank line between tools

    return "\n".join(tool_lines).rstrip()


def format_dataset_schemas(datasets: Dict[str, pd.DataFrame]) -> str:
    """
    Format dataset schemas for code generation prompts.

    Provides detailed information about each dataset including dimensions,
    column names, data types, and sample statistics. This gives the LLM
    enough context to generate correct pandas code.

    Args:
        datasets: Dictionary mapping dataset aliases to DataFrame objects

    Returns:
        Formatted string with schema information for all datasets

    Example output:
        ```
        Dataset: sales
        Rows: 15,420 | Columns: 8
        Schema:
          - order_id (int64)
          - customer_id (int64)
          - product_name (object)
          - category (object)
          - price (float64)
          - quantity (int64)
          - revenue (float64)
          - order_date (datetime64[ns])

        Dataset: customers
        Rows: 1,250 | Columns: 5
        Schema:
          - customer_id (int64)
          - name (object)
          - segment (object)
          - country (object)
          - signup_date (datetime64[ns])
        ```
    """
    if not datasets:
        return "No datasets available."

    schema_lines = []
    for alias, df in datasets.items():
        # Dataset header
        schema_lines.append(f"Dataset: {alias}")
        schema_lines.append(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        schema_lines.append("Schema:")

        # Column details
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema_lines.append(f"  - {col} ({dtype})")

        schema_lines.append("")  # Blank line between datasets

    return "\n".join(schema_lines).rstrip()


def format_loaded_datasets(datasets: Dict[str, pd.DataFrame]) -> str:
    """
    Format a summary of loaded datasets for agent context.

    Provides a concise list of available datasets with basic statistics.
    This is injected into the agent system prompt so it knows what data
    is currently available.

    Args:
        datasets: Dictionary mapping dataset aliases to DataFrame objects

    Returns:
        Formatted string listing loaded datasets with dimensions

    Example output:
        ```
        Currently loaded datasets:
          • sales (15,420 rows, 8 columns)
          • customers (1,250 rows, 5 columns)
        ```
    """
    if not datasets:
        return "Currently loaded datasets:\n  (none)"

    dataset_lines = ["Currently loaded datasets:"]
    for alias, df in datasets.items():
        rows = len(df)
        cols = len(df.columns)
        dataset_lines.append(f"  • {alias} ({rows:,} rows, {cols} columns)")

    return "\n".join(dataset_lines)


def build_agent_prompt(
    tools: List[BaseTool],
    datasets: Dict[str, pd.DataFrame],
    max_iterations: int = 20
) -> str:
    """
    Build the complete agent system prompt with injected context.

    Combines the base agent system prompt template with current tool
    descriptions, loaded dataset information, and iteration budget.

    Args:
        tools: List of available tools
        datasets: Currently loaded datasets
        max_iterations: Maximum iterations available to agent (for budget awareness)

    Returns:
        Complete system prompt ready for LLM
    """
    tool_descriptions = format_tool_descriptions(tools)
    loaded_datasets = format_loaded_datasets(datasets)

    return AGENT_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        loaded_datasets=loaded_datasets,
        max_iterations=max_iterations
    )


def build_code_generation_prompt(
    query: str,
    datasets: Dict[str, pd.DataFrame]
) -> str:
    """
    Build the code generation prompt with query and dataset context.

    Args:
        query: Natural language query to convert to pandas code
        datasets: Available datasets with schemas

    Returns:
        Complete code generation prompt ready for LLM
    """
    dataset_schemas = format_dataset_schemas(datasets)

    return CODE_GENERATION_PROMPT.format(
        query=query,
        dataset_schemas=dataset_schemas
    )


def build_visualization_prompt(
    query: str,
    datasets: Dict[str, pd.DataFrame],
    chart_type: str = "auto"
) -> str:
    """
    Build the visualization code generation prompt.

    Args:
        query: Natural language query describing desired visualization
        datasets: Available datasets with schemas
        chart_type: Suggested chart type ("auto", "line", "bar", "scatter", etc.)

    Returns:
        Complete visualization prompt ready for LLM
    """
    dataset_schemas = format_dataset_schemas(datasets)

    # Format chart type hint
    if chart_type == "auto":
        chart_hint = "Auto-detect the most appropriate chart type based on the data and query"
    else:
        chart_hint = f"Suggested chart type: {chart_type}"

    return VISUALIZATION_PROMPT.format(
        query=query,
        dataset_schemas=dataset_schemas,
        chart_type=chart_hint
    )


def build_error_retry_prompt(
    query: str,
    failed_code: str,
    error_message: str
) -> str:
    """
    Build the error retry prompt for code correction.

    Args:
        query: Original query that the code was trying to solve
        failed_code: The code that caused an error
        error_message: The error message from execution

    Returns:
        Complete error retry prompt ready for LLM
    """
    return ERROR_RETRY_PROMPT.format(
        query=query,
        failed_code=failed_code,
        error_message=error_message
    )


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_tool_response(response: Dict[str, Any]) -> bool:
    """
    Validate that a tool response from the LLM has the required structure.

    Args:
        response: Parsed JSON response from the agent

    Returns:
        True if valid, False otherwise
    """
    required_keys = {"thought", "action", "parameters"}

    if not isinstance(response, dict):
        return False

    if not all(key in response for key in required_keys):
        return False

    if not isinstance(response["thought"], str):
        return False

    if not isinstance(response["action"], str):
        return False

    if not isinstance(response["parameters"], dict):
        return False

    return True


def extract_code_from_response(response: str) -> str:
    """
    Extract clean Python code from LLM response.

    Handles cases where the LLM returns code in markdown fences or
    includes explanatory text alongside the code.

    Args:
        response: Raw response from LLM

    Returns:
        Cleaned executable Python code
    """
    # Remove markdown code fences if present
    if "```python" in response:
        # Extract content between ```python and ```
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end != -1:
            response = response[start:end]
    elif "```" in response:
        # Extract content between ``` and ```
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            response = response[start:end]

    # Strip whitespace
    code = response.strip()

    return code
