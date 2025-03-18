
from dataclasses import asdict

from dash import (
    dash_table, html, callback, MATCH, Input, State, Output, ctx, dcc,
    no_update
)
import dash_bootstrap_components as dbc

from sstatics.core import Node


class Store(dcc.Store):

    def __init__(self, name):
        super().__init__(
            id={'type': 'store', 'index': name}, storage_type='session',
            data={}
        )


class Card(dbc.Card):

    def __init__(
        self, header: str, tab_names: list[str], tab_children: list[html.Div]
    ):
        children = [
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col(html.H5(header), width=11)
                ]),
                # dbc.Tabs
            ]),
            dbc.CardBody(
                dbc.Container(id={'type': 'tab-content', 'index': header}),
                style={'height': '320px'}
            ),
        ]
        super().__init__(children=children)


class Table(html.Div):

    def __init__(self, name: str, columns: list, editable: bool = True):
        self.dash_tbl = dash_table.DataTable(
            id={'type': 'table', 'index': name},
            data=[],
            columns=columns,
            editable=editable
        )
        children = [self.dash_tbl]
        if editable:
            children.append(
                html.Button('Add row', id={'type': 'add-row', 'index': name}),
            )
        super().__init__(children)


@callback(
    Output({'type': 'table', 'index': MATCH}, 'data', allow_duplicate=True),
    Output({'type': 'store', 'index': MATCH}, 'data'),
    Input({'type': 'add-row', 'index': MATCH}, 'n_clicks'),
    State({'type': 'store', 'index': MATCH}, 'data'),
    State({'type': 'table', 'index': MATCH}, 'columns'),
    State({'type': 'table', 'index': MATCH}, 'data'),
    prevent_initial_call=True, allow_duplicate=True,
)
def add_table_row(_, stored_data, table_columns, table_data):
    new_obj_id = len(stored_data) + 1
    # Default objects that get added when the "Add row" button is clicked.
    if ctx.triggered_id['index'] == 'nodes':
        obj = Node(0, 0)
    else:
        raise ValueError(f"Unknown table name: '{ctx.triggered_id['index']}'.")
    stored_data[new_obj_id] = asdict(obj)
    new_table_entry = {'id': new_obj_id}
    for column in table_columns[1:]:  # Skip the id column.
        new_table_entry[column['id']] = getattr(obj, column['id'])
    table_data.append(new_table_entry)
    return table_data, stored_data


@callback(
    Output({'type': 'store', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'table', 'index': MATCH}, 'data'),
    State({'type': 'table', 'index': MATCH}, 'active_cell'),
    State({'type': 'store', 'index': MATCH}, 'data'),
    prevent_initial_call=True,
)
def edit_table_row(table_data, active_cell, stored_data):
    if active_cell is None:
        return no_update
    # Decide which class to use based on the name of the table.
    if ctx.triggered_id['index'] == 'nodes':
        cls = Node
    else:
        raise ValueError(f"Unknown table name: '{ctx.triggered_id['index']}'.")
    row = table_data[active_cell['row']]
    obj_id = str(row.pop('id'))
    obj_kwargs = stored_data[obj_id] | row
    # TODO: This could raise errors because of invalid arguments.
    # TODO: How to deal with this?
    obj = cls(**obj_kwargs)
    stored_data[obj_id] = asdict(obj)
    return stored_data


node_input_table = Table(
    'nodes',
    [
        {
            'name': 'Nr.',
            'id': 'id',
            'type': 'numeric',
            'editable': False

        },
        {
            'name': 'x\u0303',
            'id': 'x',
            'type': 'numeric',
            'on_change': {'failure': 'default'},
            'validation': {'default': 0},
        },
        {
            'name': 'z\u0303',
            'id': 'z',
            'type': 'numeric',
            'on_change': {'failure': 'default'},
            'validation': {'default': 0},
        }
    ]
)
