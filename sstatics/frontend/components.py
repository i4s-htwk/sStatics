
from base64 import b64decode, b64encode
import pickle

from dash import (
    dash_table, html, callback, MATCH, Input, State, Output, ctx, dcc
)
import dash_bootstrap_components as dbc

from sstatics.core import Node


class Store(dcc.Store):

    def __init__(self, name):
        super().__init__(
            id={'type': 'store', 'index': name}, storage_type='session'
        )
        self._data = None
        self.data = {}
        self.available_properties.append('test')
        self.test = 'hallo'

    @property
    def data(self):
        pickled = b64decode(self._data.encode())
        return pickle.loads(pickled)

    @data.setter
    def data(self, value):
        pickled = pickle.dumps(value)
        self._data = b64encode(pickled).decode()


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
    Output({'type': 'table', 'index': MATCH}, 'data'),
    Output({'type': 'store', 'index': MATCH}, 'data'),
    Input({'type': 'add-row', 'index': MATCH}, 'n_clicks'),
    State({'type': 'store', 'index': MATCH}, 'test'),
    State({'type': 'table', 'index': MATCH}, 'columns'),
    prevent_initial_call=True
)
def add_row(_, obj_dict, columns):
    obj = None
    new_obj_id = len(obj_dict) + 1
    if ctx.triggered_id['index'] == 'nodes':
        obj = Node()
    obj_dict[new_obj_id] = obj
    rows = [
        {c['id']: getattr(obj, c['id']) for c in columns}
        for obj in obj_dict.values()
    ]
    return rows, obj_dict


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
        },
        {
            'name': 'z\u0303',
            'id': 'z',
            'type': 'numeric',
        }
    ]
)
