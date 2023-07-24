#---------------------------------------------------------------------------------------------------------------
# Sales Dashboard
# 
# Pablo Marchesi
# Jul 2023
#---------------------------------------------------------------------------------------------------------------

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats

xlsx = r'C:\Users\Usuario\Desktop\Solid2023\Sales_Data.xlsx'
dta = pd.read_excel(xlsx)
pdp_prob = 0.01

# Acumulados por cada referencia
refs = dta['Ref. producto'].drop_duplicates().reset_index(drop = True)
cumul = pd.DataFrame({'Referencias':refs, 'Acumulado':0})

for i in range(len(refs)):
    for k in range(len(dta)):
        if dta['Ref. producto'][k] == cumul['Referencias'][i]: cumul['Acumulado'][i] = cumul['Acumulado'][i] + dta['Cant.'][k]

cumul = cumul.sort_values(by = ['Acumulado'], ascending = False )


# Pedidos por semanas y producto 
df = pd.DataFrame()
df['Semana'] = pd.to_datetime(dta['Fecha de creación'])
df.set_index('Semana', inplace=True)
df = df.resample('W').sum()
weeks = df.index
df['Periodo'] = weeks.to_period('W')
df = df.reset_index(drop = True)


for i in range(len(refs)):
    ref = dta[dta['Ref. producto'] == refs[i]].reset_index()
    dt = list(ref['Fecha de creación'])
    df[refs[i]] = 0

    for k in range(len(dt)):
        for j in range(len(weeks)):
            if dt[k].to_period('W') == df['Periodo'][j]: df.loc[j, refs[i]] += ref['Cant.'][k]             

df['Periodo'] = weeks
df = df.set_index('Periodo')
df['Totales'] = df.sum(axis = 1)
df = df.sort_index(axis=1, ascending= False)


df_m = df.resample('M').sum()
refs.name = 'Nº Pedidos'
x = np.arange(0, 20)
poi = pd.DataFrame(columns = refs)

for l in range(len(refs)):
    var = list(df_m[refs[l]])
    lmbda = np.mean(var)
    fit = stats.poisson(lmbda)
    poi[refs[l]] = fit.pmf(x)

poi['Totales'] = 0

# Función auxiliar para obtener la suma de los valores de la variable seleccionada
def get_variable_sum(variable):
    return df[variable].sum()


# Crear la aplicación Dash
app = dash.Dash(__name__)

# Opciones para la lista desplegable
dropdown_options = [{'label': col, 'value': col} for col in df.columns]

# Definir el estilo para la lista desplegable
dropdown_style = {'position': 'absolute', 'top': '12px', 'right': '300px', 'width': '200px'}

# Definir el diseño del dashboard
app.layout = html.Div(
    children=[
        html.H1('Dashboard SolidSoft'),
        
        html.Div(
            children=[
                dcc.Graph(
                    id='Pedidos',
                    figure=px.line(df, title='Pedidos semanales de {}'.format(refs[0]))
                ),
                dcc.Graph(
                    id='Probabilidad pedidos',
                    figure=px.line(poi, title='Probabilidad pedidos mensuales'),
                ),
                dcc.Dropdown(
                    id='variable-dropdown',
                    options=dropdown_options,
                    value=refs[0],  # Valor inicial de la lista desplegable
                    style=dropdown_style
                ),
                html.P(id='variable-sum')  # Línea de texto para mostrar la suma de los valores
            ],
            style={'position': 'relative'}
        )
    ],
    # style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'}
)


# Actualizar los gráficos al seleccionar una variable en la lista desplegable
@app.callback(
    dash.dependencies.Output('Pedidos', 'figure'),
    dash.dependencies.Output('Probabilidad pedidos', 'figure'),
    dash.dependencies.Output('variable-sum', 'children'),  # Actualizar el texto de la suma de valores
    [dash.dependencies.Input('variable-dropdown', 'value')]
)
def update_graph(variable):
    variable_sum = get_variable_sum(variable)
    fig_pedidos = px.line(df, y=variable, title='Pedidos semanales de {} - Total periodo: {}'.format(variable, variable_sum))
    fig_poi = px.line(poi, x=x, y=variable, range_x=[0, 20], range_y=[0,max(poi[variable])+0.05],markers=True, title='Probabilidad pedidos mensuales de {}'.format(variable),
                      labels={'x': 'Nº Pedidos', 'y': 'Probabilidad', 'color': 'Referencia'})

    # Obtener el primer índice donde el valor sea menor que 0.01
    index_value = (poi[variable] < pdp_prob).idxmax()
    # Obtener el valor x correspondiente al primer índice menor que 0.01 desde el lado derecho
    x_value = x[index_value]

    # Agregar la línea vertical punteada al layout del gráfico
    # fig_poi.update_layout(shapes=[
    #     dict(
    #         type='line',
    #         x0=x_value,
    #         y0=0,
    #         x1=x_value,
    #         y1=max(poi[variable])+0.05,
    #         line=dict(
    #             color='black',
    #             width=2,
    #             dash='dot'
    #         )
    #     )
    # ])
    # # Agregar texto a la línea vertical punteada
    # fig_poi.add_annotation(
    #     x=x_value + 0.75,
    #     y=max(poi[variable])/2,
    #     text='Punto de Pedido 1%',
    #     showarrow=False,
    #     font=dict(
    #         size=12,
    #         color='black'
    #     )
    # )

    return fig_pedidos, fig_poi, f""


# Ejecutar la aplicación Dash
if __name__ == '__main__':
    app.run_server(debug=True)
