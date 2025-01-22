import sys
import io
from code import InteractiveConsole
from multiprocessing import Queue
from typing import Any


class CapturingInteractiveConsole(InteractiveConsole):
    def __init__(self, locals=None):
        super().__init__(locals=locals)
        self.output_buffer = io.StringIO()
    
    def runcode(self, code):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = self.output_buffer
        sys.stderr = self.output_buffer

        try:
            super().runcode(code)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def get_output(self):
        val = self.output_buffer.getvalue()
        self.output_buffer.seek(0)
        self.output_buffer.truncate(0)
        return val.rstrip("\n")

def repl_worker(input_queue: Queue, output_queue: Queue, local_context: dict[str, Any]):
    console = CapturingInteractiveConsole(locals=local_context)
    
    # Define the utility functions, including `ts_plot_unique`
    console.runcode(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "import json\n"
        "\n"
        "def ts_plot_unique(df, nome, source, units, chart):\n"
        "    fig = go.Figure()\n"
        "    colors = ['#0A3254', '#B2292E', '#E0D253', '#7AADD4', '#336094', '#FAAF90', '#054FB9']\n"
        "    if len(df.columns) == 2:\n"
        "        colors = ['#219ebc', '#023047']\n"
        "    elif len(df.columns) == 3:\n"
        "        colors = ['#8ecae6', '#219ebc', '#ffb703', '#fb8500', '#023047']\n"
        "    elif len(df.columns) == 4:\n"
        "        colors = ['#8ecae6', '#219ebc', '#ffb703', '#fb8500', '#126782', '#023047']\n"
        "    elif len(df.columns) == 5:\n"
        "        colors = ['#8ecae6', '#219ebc', '#023047', '#ffb703', '#F0A688', '#EC7A8F', '#fd9e02']\n"
        "    else:\n"
        "        colors = ['#8ecae6', '#219ebc', '#126782', '#023047', '#ffb703', '#fd9e02', '#fb8500']\n"
        "\n"
        "    if chart == 'Bar':\n"
        "        for i in range(len(df.columns)):\n"
        "            fig.add_trace(go.Bar(x=df.index, y=df.iloc[:, i], marker=dict(color=colors[i]), name=str(df.columns[i])))\n"
        "    elif chart == 'Scatter':\n"
        "        fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], mode='markers', marker=dict(color=colors[1])))\n"
        "    elif chart == 'stacked_area':\n"
        "        for i in range(len(df.columns)):\n"
        "            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, i], mode='lines', stackgroup='one', name=str(df.columns[i]), marker=dict(color=colors[i])))\n"
        "    else:\n"
        "        for i in range(len(df.columns)):\n"
        "            fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, i], line=dict(color=colors[i], width=3), name=str(df.columns[i])))\n"
        "\n"
        "    fig.update_layout(\n"
        "        title={'text': '<b>' + nome + '<b>', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'bottom'},\n"
        "        paper_bgcolor='rgba(250,250,250)',\n"
        "        plot_bgcolor='rgba(0,0,0,0)',\n"
        "        font_size=12,\n"
        "        font_color='#0D1018',\n"
        "        yaxis_title=units,\n"
        "        template='plotly_white',\n"
        "        font_family='Verdana',\n"
        "        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=-0.01),\n"
        "        autosize=True,\n"
        "        height=700\n"
        "    )\n"
        "    return fig.to_json()\n"
        "\n"
        "def return_chart_custom(df, nome, source, units, chart):\n"
        "    chart_json = ts_plot_unique(df, nome, source, units, chart)\n"
        "    chart_data = {\n"
        "        'type': 'chart',\n"
        "        'content': chart_json\n"
        "    }\n"
        "    return '```json\\n' + json.dumps(chart_data) + '\\n```'\n"
    )

    # REPL loop
    while True:
        code = input_queue.get()
        if code is None:
            break
        try:
            lines = code.strip().split("\n")
            if len(lines) > 1:
                stmt_code = "\n".join(lines[:-1])
                if stmt_code:
                    compiled_stmt = compile(stmt_code, "<input>", "exec")
                    console.runcode(compiled_stmt)
                try:
                    compiled_expr = compile(lines[-1], "<input>", "eval")
                    result = eval(compiled_expr, console.locals)
                    if result is not None:
                        print(result, file=console.output_buffer)
                except SyntaxError:
                    compiled_stmt = compile(lines[-1], "<input>", "exec")
                    console.runcode(compiled_stmt)
            else:
                try:
                    if "print(" in code:
                        compiled_stmt = compile(code, "<input>", "exec")
                        console.runcode(compiled_stmt)
                    else:
                        compiled_expr = compile(code, "<input>", "eval")
                        result = eval(compiled_expr, console.locals)
                        if result is not None:
                            print(result, file=console.output_buffer)
                except SyntaxError:
                    compiled = compile(code, "<input>", "eval")
                    result = eval(compiled, console.locals)
                    if result is not None:
                        print(result, file=console.output_buffer)
            output = console.get_output()
            output_queue.put(output)
        except Exception as e:
            output_queue.put(str(e))

# def repl_worker(input_queue: Queue, output_queue: Queue, local_context: dict[str, Any]):
#     console = CapturingInteractiveConsole(locals=local_context)
#     Load our utility functions that the LLM can use to return structured data
#     console.runcode(
#         "import numpy as np\n"
#         "import pandas as pd\n"
#         "import json\n"
#         "\n"
#         "def return_structured(data):\n"
#         "    table_data = {\n"
#         '        "type": "table",\n'
#         "    }\n"
#         "    if isinstance(data, pd.DataFrame):\n"
#         '        table_data["content"] = data.to_json(orient="records", date_format="iso")\n'
#         "    elif isinstance(data, pd.Series):\n"
#         '        table_data["content"] = data.to_json(orient="records", date_format="iso")\n'
#         "    elif isinstance(data, np.ndarray):\n"
#         "        df = pd.DataFrame(data)\n"
#         '        table_data["content"] = df.to_json(orient="records", date_format="iso")\n'
#         "    return '```json\\n' + json.dumps(table_data) + '\\n```'\n"
#         "\n"
#         "def return_chart(df, chart_type, xKey, yKey):\n"
#         "    chart_data = {\n"
#         '        "type": "chart",\n'
#         '        "content": df.to_json(orient="records", date_format="iso"),\n'
#         '        "chart_params": {\n'
#         '            "chartType": chart_type,\n'
#         '            "xKey": xKey,\n'
#         '            "yKey": yKey\n'
#         "        }\n"
#         "    }\n"
#         "    return '```json\\n' + json.dumps(chart_data) + '\\n```'\n"
#     )
    
#     # Begin the REPL loop
#     while True:
#         code = input_queue.get()
#         if code is None:
#             break
#         try:
#             lines = code.strip().split("\n")
#             if len(lines) > 1:
#                 # Execute all but the last line as statements
#                 stmt_code = "\n".join(lines[:-1])
#                 if stmt_code:
#                     compiled_stmt = compile(stmt_code, "<input>", "exec")
#                     console.runcode(compiled_stmt)

#                 # Try to evaluate the last line for its result
#                 try:
#                     compiled_expr = compile(lines[-1], "<input>", "eval")
#                     result = eval(compiled_expr, console.locals)
#                     if result is not None:
#                         print(result, file=console.output_buffer)
#                 except SyntaxError:
#                     compiled_stmt = compile(lines[-1], "<input>", "exec")
#                     console.runcode(compiled_stmt)
#             else:
#                 # Single line - try as statement first, then expression
#                 try:
#                     # Not pretty, but a quick way to handle the rare case where
#                     # we get a print statement as a single line (since it can
#                     # also get evaluated as an expression)
#                     if "print(" in code:
#                         compiled_stmt = compile(code, "<input>", "exec")
#                         console.runcode(compiled_stmt)
#                     else:
#                         compiled_expr = compile(code, "<input>", "eval")
#                         result = eval(compiled_expr, console.locals)
#                         if result is not None:
#                             print(result, file=console.output_buffer)
#                 except SyntaxError:
#                     # If it's not a valid statement, try as expression
#                     compiled = compile(code, "<input>", "eval")
#                     result = eval(compiled, console.locals)
#                     if result is not None:
#                         print(result, file=console.output_buffer)

#             output = console.get_output()
#             output_queue.put(output)
#         except Exception as e:
            output_queue.put(str(e))
