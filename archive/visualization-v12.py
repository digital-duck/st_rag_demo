# Location: utils/visualization.py
# Visualization utilities for RAG applications

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def generate_visualization(data, chart_type="auto"):
    """
    Generate a Plotly visualization based on data and chart type
    
    Args:
        data: Pandas DataFrame containing the data to visualize
        chart_type (str): Type of chart to generate (auto, bar, line, scatter, pie, table)
        
    Returns:
        plotly.graph_objects.Figure or str: The visualization or error message
    """
    if isinstance(data, str):
        return data  # Return error message if data is a string
    
    try:
        # Determine the best chart type if auto
        if chart_type == "auto":
            num_columns = sum(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns)
            categorical_columns = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
            
            if len(data) > 20 and num_columns >= 2:
                chart_type = "scatter"
            elif len(categorical_columns) >= 1 and num_columns >= 1:
                chart_type = "bar"
            elif num_columns >= 1:
                chart_type = "line"
            else:
                chart_type = "table"
        
        # Create visualization based on chart type
        if chart_type == "scatter" and len(data.columns) >= 2:
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            if len(numeric_cols) >= 2:
                fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1])
                return fig
        
        elif chart_type == "bar" and len(data.columns) >= 2:
            # Find a categorical column and a numeric column
            categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            
            if categorical_cols and numeric_cols:
                fig = px.bar(data, x=categorical_cols[0], y=numeric_cols[0])
                return fig
        
        elif chart_type == "line" and len(data.columns) >= 2:
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            if len(numeric_cols) >= 1:
                # Use first column as x if it's datetime or the index
                if pd.api.types.is_datetime64_any_dtype(data.index):
                    fig = px.line(data, y=numeric_cols[0])
                else:
                    fig = px.line(data, y=numeric_cols[0], x=data.columns[0])
                return fig
        
        elif chart_type == "pie" and len(data.columns) >= 2:
            # Find a categorical column and a numeric column
            categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
            numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            
            if categorical_cols and numeric_cols:
                fig = px.pie(data, names=categorical_cols[0], values=numeric_cols[0])
                return fig
        
        # Default to table view
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(data.columns)),
            cells=dict(values=[data[col] for col in data.columns])
        )])
        return fig
        
    except Exception as e:
        return f"Error generating visualization: {str(e)}"

def generate_visualization_code(df, chart_type):
    """
    Generate Python code that would create the visualization
    
    Args:
        df: pandas DataFrame with the data
        chart_type: Type of chart to generate (bar, line, scatter, pie)
        
    Returns:
        str: Python code snippet
    """
    # Get column information
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Create sample code snippet
    code = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
    
    # Add pandas DataFrame creation code with sample data
    code += "\n# Create DataFrame with your data\ndf = pd.DataFrame({\n"
    for col in columns:
        sample_values = str(df[col].head(3).tolist()).replace('[', '').replace(']', '')
        if pd.api.types.is_numeric_dtype(df[col]):
            code += f"    '{col}': [{sample_values}, ...],\n"
        else:
            code += f"    '{col}': ['{sample_values}', ...],\n"
    code += "})\n\n"
    
    # Add visualization code based on chart type
    if chart_type == "bar" and len(categorical_cols) > 0 and len(numeric_cols) > 0:
        x_col = categorical_cols[0]
        y_col = numeric_cols[0]
        
        code += f"""# Create bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='{x_col}', y='{y_col}', data=df)
plt.title('{y_col} by {x_col}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""

    elif chart_type == "line" and len(numeric_cols) > 0:
        y_col = numeric_cols[0]
        x_col = columns[0] if columns[0] != y_col else columns[1] if len(columns) > 1 else "index"
        
        if x_col == "index":
            code += f"""# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(df['{y_col}'])
plt.title('{y_col} over Index')
plt.ylabel('{y_col}')
plt.xlabel('Index')
plt.grid(True)
plt.tight_layout()
plt.show()"""
        else:
            code += f"""# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(df['{x_col}'], df['{y_col}'])
plt.title('{y_col} vs {x_col}')
plt.ylabel('{y_col}')
plt.xlabel('{x_col}')
plt.grid(True)
plt.tight_layout()
plt.show()"""

    elif chart_type == "scatter" and len(numeric_cols) >= 2:
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        code += f"""# Create scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='{x_col}', y='{y_col}', data=df)
plt.title('{y_col} vs {x_col}')
plt.tight_layout()
plt.show()"""

    elif chart_type == "pie" and len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat_col = categorical_cols[0]
        value_col = numeric_cols[0]
        
        code += f"""# Create pie chart
plt.figure(figsize=(10, 6))
df_grouped = df.groupby('{cat_col}')['{value_col}'].sum()
plt.pie(df_grouped, labels=df_grouped.index, autopct='%1.1f%%')
plt.title('{value_col} Distribution by {cat_col}')
plt.axis('equal')
plt.tight_layout()
plt.show()"""
        
    else:
        # Default to a generic histogram of the first numeric column
        if numeric_cols:
            col = numeric_cols[0]
            code += f"""# Create histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['{col}'], kde=True)
plt.title('Distribution of {col}')
plt.tight_layout()
plt.show()"""
        else:
            # Fallback if no numeric columns
            code += """# No numeric columns found for visualization
# Display the data instead
print(df)"""
    
    return code