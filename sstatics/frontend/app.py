
import dash
from dash import html

from sstatics.frontend.components import node_input_table, Store

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        node_input_table,
        Store(name='nodes')
    ]
)

if __name__ == '__main__':
    app.run(debug=True)