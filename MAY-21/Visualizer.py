import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
from DataFetcher import fetch_csv_data
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_column_suggestions(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Use Gemini to suggest appropriate x and y columns for visualization.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze
        
    Returns:
        Tuple[str, str]: Suggested x_column and y_column names
    """
    # Create a description of the DataFrame
    df_info = f"""
    DataFrame columns: {', '.join(df.columns)}
    Column types: {df.dtypes.to_dict()}
    Sample data:
    {df.head().to_string()}
    """
    
    prompt = f"""Given the following DataFrame information:
        {df_info}

        Please analyze the DataFrame schema and suggest the most meaningful and relevant columns for creating an insightful visualization. Choose columns that are likely to show interesting patterns, trends, or comparisons.

        Guidelines:
        1. For the x-axis:
        - Prefer columns that represent categories, timestamps, or ordered sequences.
        - Choose a column that provides clear separation or grouping when visualized.

        2. For the y-axis:
        - Select a column with numeric values that vary meaningfully across the x-axis.
        - Avoid columns with low variance or those unlikely to convey actionable insights.

        3. Prioritize relevance:
        - Choose x and y combinations that are semantically related (e.g., sales over time, scores per category).
        - Avoid IDs, primary keys, or redundant columns unless they are meaningful in context.

        Return your response in the following format:
        x_column: column_name
        y_column: column_name
        """

    
    try:
        response = model.generate_content(prompt)
        # Parse the response to extract column names
        lines = response.text.strip().split('\n')
        x_col = lines[0].split(': ')[1].strip()
        y_col = lines[1].split(': ')[1].strip()
        
        # Validate that the suggested columns exist
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Suggested columns not found in DataFrame")
            
        return x_col, y_col
    except Exception as e:
        print(f"Error getting column suggestions: {str(e)}")
        # Fallback to default behavior
        return df.index.name or "index", df.select_dtypes(include=['number']).columns[0]

def visualize_csv_data(csv_path: str, x_column: Optional[str] = None, y_column: Optional[str] = None, plot_type: str = "line", use_ai_suggestions: bool = True) -> dict:
    """
    Visualize CSV data with AI-powered column selection.
    
    Args:
        csv_path (str): Path to the CSV file
        x_column (Optional[str]): Column name for x-axis
        y_column (Optional[str]): Column name for y-axis
        plot_type (str): Type of plot ('line', 'bar', or 'scatter')
        use_ai_suggestions (bool): Whether to use AI for column selection
        
    Returns:
        dict: Dictionary containing plot information and path
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # If use_ai_suggestions is True, get suggestions from Gemini
    if use_ai_suggestions:
        x_column, y_column = get_column_suggestions(df)
    
    # If no columns specified, use index for x and first numeric column for y
    if x_column is None:
        x_column = df.index
    if y_column is None:
        y_column = df.select_dtypes(include=['number']).columns[0]
    
    plt.figure(figsize=(10, 6))
    
    if plot_type == "line":
        plt.plot(df[x_column], df[y_column])
    elif plot_type == "bar":
        plt.bar(df[x_column], df[y_column])
    elif plot_type == "scatter":
        plt.scatter(df[x_column], df[y_column])
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    plt.title(f"{plot_type.capitalize()} Plot of {y_column} vs {x_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)

    output_dir = "MAY-21/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = f"{output_dir}/{plot_type}_{y_column}_vs_{x_column}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return {
        "plot_type": plot_type,
        "x_column": x_column,
        "y_column": y_column,
        "plot_path": plot_path
    }

def visualize_statistics(csv_path: str) -> dict:
    """
    Create statistical visualizations (mean, median, variance) for numerical columns.
    """
    # Read the CSV file
    data = pd.read_csv(csv_path)
    
    # Get numerical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numerical_cols) == 0:
        return {"error": "No numerical columns found in the data"}
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Calculate statistics
    stats = {
        'Mean': data[numerical_cols].mean(),
        'Median': data[numerical_cols].median(),
        'Variance': data[numerical_cols].var()
    }
    
    # Create bar plots for each statistic
    for idx, (stat_name, stat_values) in enumerate(stats.items()):
        ax = axes[idx]
        stat_values.plot(kind='bar', ax=ax)
        ax.set_title(f'{stat_name} of Numerical Columns')
        ax.set_xlabel('Columns')
        ax.set_ylabel(stat_name)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = "statistical_analysis.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Create a summary of statistics
    summary = {
        "statistics": {
            "mean": stats['Mean'].to_dict(),
            "median": stats['Median'].to_dict(),
            "variance": stats['Variance'].to_dict()
        },
        "plot_path": plot_path,
        "numerical_columns": list(numerical_cols)
    }
    
    return summary

