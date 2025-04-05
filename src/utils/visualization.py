# Location: utils/visualization.py
# Visualization utilities for RAG applications

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime
import re

def generate_visualization(data, chart_type="auto", x_column=None, y_column=None, color_column=None):
    """
    Generate a Plotly visualization based on data and chart type
    
    Args:
        data: Pandas DataFrame containing the data to visualize
        chart_type (str): Type of chart to generate (auto, bar, line, scatter, pie, table)
        x_column (str, optional): Column to use for x-axis
        y_column (str, optional): Column to use for y-axis
        color_column (str, optional): Column to use for color
        
    Returns:
        plotly.graph_objects.Figure or str: The visualization or error message
    """
    if isinstance(data, str):
        return data  # Return error message if data is a string
    
    # Check if DataFrame is empty
    if data.empty:
        return "No data to visualize"
    
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Auto-select columns if not provided
        if x_column is None or y_column is None:
            num_columns = sum(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
            categorical_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
            numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            # Determine best chart type if auto
            if chart_type == "auto":
                if len(df) > 20 and num_columns >= 2:
                    chart_type = "scatter"
                elif len(categorical_columns) >= 1 and num_columns >= 1:
                    chart_type = "bar"
                elif num_columns >= 1:
                    chart_type = "line"
                else:
                    chart_type = "table"
            
            # Auto-select x_column if not provided
            if x_column is None:
                if categorical_columns and (chart_type in ["bar", "pie"]):
                    x_column = categorical_columns[0]
                elif numeric_columns and chart_type in ["scatter", "line"]:
                    x_column = numeric_columns[0]
                elif df.columns.size > 0:
                    x_column = df.columns[0]
            
            # Auto-select y_column if not provided
            if y_column is None:
                if numeric_columns:
                    for col in numeric_columns:
                        if col != x_column:
                            y_column = col
                            break
                    if y_column is None and numeric_columns:
                        y_column = numeric_columns[0]
                elif df.columns.size > 1:
                    y_column = df.columns[1]
                else:
                    y_column = df.columns[0]
        
        # Create visualization based on chart type
        if chart_type == "scatter":
            # Create the scatter chart
            if color_column and color_column in df.columns:
                fig = px.scatter(
                    df, x=x_column, y=y_column, color=color_column,
                    title=f"Scatter Plot: {y_column} vs {x_column}",
                    labels={x_column: x_column.replace('_', ' ').title(), 
                           y_column: y_column.replace('_', ' ').title()}
                )
            else:
                fig = px.scatter(
                    df, x=x_column, y=y_column,
                    title=f"Scatter Plot: {y_column} vs {x_column}",
                    labels={x_column: x_column.replace('_', ' ').title(), 
                           y_column: y_column.replace('_', ' ').title()}
                )
            
            # Add reference line for clarity
            if (pd.api.types.is_numeric_dtype(df[x_column]) and 
                pd.api.types.is_numeric_dtype(df[y_column])):
                
                # Calculate trendline
                z = np.polyfit(df[x_column], df[y_column], 1)
                p = np.poly1d(z)
                
                # Add trendline to plot
                fig.add_trace(go.Scatter(
                    x=[df[x_column].min(), df[x_column].max()],
                    y=[p(df[x_column].min()), p(df[x_column].max())],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend',
                    showlegend=True
                ))
                
            return fig
        
        elif chart_type == "bar":
            # Check for too many categories
            if pd.api.types.is_object_dtype(df[x_column]) and df[x_column].nunique() > 30:
                # If too many categories, take top 20 by value
                top_categories = df.groupby(x_column)[y_column].sum().nlargest(20).index
                df = df[df[x_column].isin(top_categories)]
            
            # Create the bar chart
            if color_column and color_column in df.columns:
                fig = px.bar(
                    df, x=x_column, y=y_column, color=color_column,
                    title=f"{y_column} by {x_column}",
                    labels={x_column: x_column.replace('_', ' ').title(), 
                           y_column: y_column.replace('_', ' ').title()}
                )
            else:
                fig = px.bar(
                    df, x=x_column, y=y_column,
                    title=f"{y_column} by {x_column}",
                    labels={x_column: x_column.replace('_', ' ').title(), 
                           y_column: y_column.replace('_', ' ').title()}
                )
            
            # Add value labels on top of bars for small datasets
            if len(df) <= 15:
                fig.update_traces(texttemplate='%{y}', textposition='outside')
            
            # Improve layout for better readability
            fig.update_layout(
                xaxis_title=x_column.replace('_', ' ').title(),
                yaxis_title=y_column.replace('_', ' ').title()
            )
            
            # Rotate x-axis labels if there are many categories or long names
            if df[x_column].nunique() > 8 or df[x_column].astype(str).str.len().max() > 10:
                fig.update_layout(xaxis_tickangle=-45)
            
            return fig
        
        elif chart_type == "line":
            # Sort by x column if it's a datetime or numeric
            if pd.api.types.is_datetime64_any_dtype(df[x_column]) or pd.api.types.is_numeric_dtype(df[x_column]):
                df = df.sort_values(by=x_column)
            
            # Create the line chart
            if color_column and color_column in df.columns:
                fig = px.line(
                    df, x=x_column, y=y_column, color=color_column,
                    title=f"{y_column} over {x_column}",
                    labels={x_column: x_column.replace('_', ' ').title(), 
                           y_column: y_column.replace('_', ' ').title()}
                )
            else:
                fig = px.line(
                    df, x=x_column, y=y_column,
                    title=f"{y_column} over {x_column}",
                    labels={x_column: x_column.replace('_', ' ').title(), 
                           y_column: y_column.replace('_', ' ').title()}
                )
            
            # Add markers for small datasets
            if len(df) <= 30:
                fig.update_traces(mode='lines+markers')
            
            # Improve layout
            fig.update_layout(
                xaxis_title=x_column.replace('_', ' ').title(),
                yaxis_title=y_column.replace('_', ' ').title()
            )
            
            return fig
        
        elif chart_type == "pie":
            # Aggregate data by x_column if multiple values exist
            if df.groupby(x_column)[y_column].count().max() > 1:
                pie_data = df.groupby(x_column)[y_column].sum().reset_index()
            else:
                pie_data = df
            
            # Limit number of slices to avoid cluttered pie chart
            if pie_data[x_column].nunique() > 10:
                # Group all but the top 9 values into 'Other'
                top_values = pie_data.nlargest(9, y_column)[x_column].values
                pie_data = pd.concat([
                    pie_data[pie_data[x_column].isin(top_values)],
                    pd.DataFrame({
                        x_column: ['Other'],
                        y_column: [pie_data[~pie_data[x_column].isin(top_values)][y_column].sum()]
                    })
                ])
            
            # Create the pie chart
            fig = px.pie(
                pie_data, names=x_column, values=y_column,
                title=f"Distribution of {y_column} by {x_column}",
                labels={x_column: x_column.replace('_', ' ').title(), 
                       y_column: y_column.replace('_', ' ').title()}
            )
            
            # Improve layout
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return fig
        
        elif chart_type == "table":
            # Create a table visualization
            header_values = list(df.columns)
            cell_values = [df[col] for col in df.columns]
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=header_values,
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=cell_values,
                    fill_color='lavender',
                    align='left'
                )
            )])
            
            fig.update_layout(
                title="Data Table View",
                height=400 + min(len(df), 10) * 25  # Adjust height based on row count
            )
            
            return fig
        
        else:
            # Default to a table view if chart type is not recognized
            return generate_visualization(df, "table", x_column, y_column)
        
    except Exception as e:
        return f"Error generating visualization: {str(e)}"

def generate_visualization_code(df, chart_type, x_column=None, y_column=None, color_column=None):
    """
    Generate Python code that would create the visualization
    
    Args:
        df: pandas DataFrame with the data
        chart_type: Type of chart to generate (bar, line, scatter, pie)
        x_column (str, optional): Column to use for x-axis
        y_column (str, optional): Column to use for y-axis
        color_column (str, optional): Column to use for color
        
    Returns:
        str: Python code snippet
    """
    # Get column information
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Use specified columns or auto-select
    if x_column is None:
        if categorical_cols and chart_type in ["bar", "pie"]:
            x_column = categorical_cols[0]
        elif numeric_cols:
            x_column = numeric_cols[0]
        elif columns:
            x_column = columns[0]
    
    if y_column is None:
        if numeric_cols:
            for col in numeric_cols:
                if col != x_column:
                    y_column = col
                    break
            if y_column is None and numeric_cols:
                y_column = numeric_cols[0]
        elif len(columns) > 1:
            y_column = columns[1]
        else:
            y_column = columns[0]
    
    # Create sample code snippet
    code = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
    
    # Add pandas DataFrame creation code with sample data
    code += "\n# Create DataFrame with your data\ndf = pd.DataFrame({\n"
    for col in columns[:5]:  # Limit to first 5 columns for brevity
        sample_values = str(df[col].head(3).tolist()).replace('[', '').replace(']', '')
        if pd.api.types.is_numeric_dtype(df[col]):
            code += f"    '{col}': [{sample_values}, ...],\n"
        else:
            code += f"    '{col}': ['{sample_values}', ...],\n"
    code += "})\n\n"
    
    # Add visualization code based on chart type
    if chart_type == "bar":
        code += f"""# Create bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='{x_column}', y='{y_column}', data=df)
plt.title('{y_column} by {x_column}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Alternatively with Plotly
import plotly.express as px
fig = px.bar(df, x='{x_column}', y='{y_column}', title='{y_column} by {x_column}')
fig.show()
"""

    elif chart_type == "line":
        code += f"""# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(df['{x_column}'], df['{y_column}'])
plt.title('{y_column} vs {x_column}')
plt.ylabel('{y_column}')
plt.xlabel('{x_column}')
plt.grid(True)
plt.tight_layout()
plt.show()

# Alternatively with Plotly
import plotly.express as px
fig = px.line(df, x='{x_column}', y='{y_column}', title='{y_column} over {x_column}')
fig.show()
"""

    elif chart_type == "scatter":
        code += f"""# Create scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='{x_column}', y='{y_column}', data=df)
plt.title('{y_column} vs {x_column}')
plt.tight_layout()
plt.show()

# Alternatively with Plotly
import plotly.express as px
fig = px.scatter(df, x='{x_column}', y='{y_column}', title='Scatter Plot: {y_column} vs {x_column}')
fig.show()
"""

    elif chart_type == "pie":
        code += f"""# Create pie chart
plt.figure(figsize=(10, 6))
df_grouped = df.groupby('{x_column}')['{y_column}'].sum()
plt.pie(df_grouped, labels=df_grouped.index, autopct='%1.1f%%')
plt.title('{y_column} Distribution by {x_column}')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Alternatively with Plotly
import plotly.express as px
fig = px.pie(df, names='{x_column}', values='{y_column}', title='Distribution of {y_column} by {x_column}')
fig.show()
"""
        
    else:
        # Default to a generic histogram of the first numeric column
        if numeric_cols:
            col = numeric_cols[0]
            code += f"""# Create histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['{col}'], kde=True)
plt.title('Distribution of {col}')
plt.tight_layout()
plt.show()

# Alternatively with Plotly
import plotly.express as px
fig = px.histogram(df, x='{col}', title='Distribution of {col}')
fig.show()
"""
        else:
            # Fallback if no numeric columns
            code += """# No numeric columns found for visualization
# Display the data instead
print(df)"""
    
    return code

def suggest_visualizations(df):
    """
    Suggest appropriate visualizations based on the data
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        
    Returns:
        list: List of visualization suggestions
    """
    suggestions = []
    
    # Get column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Try to detect datetime columns
    datetime_cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                datetime_cols.append(col)
            except:
                pass
    
    # Check for time series data
    if datetime_cols:
        for date_col in datetime_cols:
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                suggestions.append({
                    'type': 'line',
                    'x_column': date_col,
                    'y_column': num_col,
                    'title': f"Time series of {num_col} over {date_col}"
                })
    
    # Check for categorical distributions
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            if df[cat_col].nunique() <= 15:  # Only if not too many categories
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    suggestions.append({
                        'type': 'bar',
                        'x_column': cat_col,
                        'y_column': num_col,
                        'title': f"Distribution of {num_col} by {cat_col}"
                    })
                    
                    if df[cat_col].nunique() <= 7:  # Only if few categories
                        suggestions.append({
                            'type': 'pie',
                            'x_column': cat_col,
                            'y_column': num_col,
                            'title': f"Proportion of {num_col} by {cat_col}"
                        })
    
    # Check for correlations between numeric columns
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):  # Limit to first 3 numeric columns
            for col2 in numeric_cols[i+1:min(i+4, len(numeric_cols))]:  # Compare with next 3 columns
                suggestions.append({
                    'type': 'scatter',
                    'x_column': col1,
                    'y_column': col2,
                    'title': f"Correlation between {col1} and {col2}"
                })
    
    # Add table view as a fallback
    suggestions.append({
        'type': 'table',
        'title': "Table view of the data"
    })
    
    return suggestions

def format_sql_for_display(sql_query):
    """
    Format SQL query for better display
    
    Args:
        sql_query (str): SQL query to format
        
    Returns:
        str: Formatted SQL query
    """
    # List of SQL keywords to capitalize
    keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
        'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
        'UNION', 'UNION ALL', 'INTERSECT', 'EXCEPT', 'LIMIT',
        'INSERT INTO', 'UPDATE', 'DELETE', 'CREATE TABLE', 'ALTER TABLE',
        'DROP TABLE', 'CREATE INDEX', 'DROP INDEX', 'AS', 'ON', 'AND', 'OR'
    ]
    
    # Capitalize SQL keywords
    formatted_sql = sql_query
    for keyword in keywords:
        # Use regex to match whole words only
        pattern = r'(?i)\b{}\b'.format(re.escape(keyword))
        formatted_sql = re.sub(pattern, keyword, formatted_sql)
    
    # Add line breaks after certain keywords
    line_break_keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
        'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
        'UNION', 'UNION ALL', 'INTERSECT', 'EXCEPT', 'LIMIT'
    ]
    
    for keyword in line_break_keywords:
        # Add line break after keyword
        pattern = r'{} '.format(re.escape(keyword))
        formatted_sql = re.sub(pattern, '{}\n    '.format(keyword), formatted_sql)
    
    return formatted_sql