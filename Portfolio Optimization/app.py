import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px

# ---------------- DATA ---------------- #
stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
data = yf.download(stocks, period="5y")

if isinstance(data.columns, pd.MultiIndex):
    data = data['Close']

returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# ---------------- FUNCTIONS ---------------- #
def portfolio_performance(weights):
    ret = np.sum(weights * mean_returns) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return ret, std

def negative_sharpe(weights):
    ret, std = portfolio_performance(weights)
    return -(ret / std)

# ---------------- OPTIMIZATION ---------------- #
num_assets = len(stocks)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for _ in range(num_assets))
init = num_assets * [1./num_assets]

opt = minimize(negative_sharpe, init, method='SLSQP',
               bounds=bounds, constraints=constraints)

weights = opt.x
opt_return, opt_risk = portfolio_performance(weights)
sharpe = opt_return / opt_risk

# ---------------- EFFICIENT FRONTIER ---------------- #
num_portfolios = 3000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    w = np.random.random(num_assets)
    w /= np.sum(w)
    r, s = portfolio_performance(w)
    results[0,i] = r
    results[1,i] = s
    results[2,i] = r/s

# ---------------- DASH APP ---------------- #
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor':'#0d1117','color':'white','padding':'20px'}, children=[

    html.H1("📊 Portfolio Optimization Dashboard", style={'textAlign':'center'}),

# ---- KPI CARDS ---- #
    html.Div([
        html.Div([html.H3("Return"), html.H2(f"{opt_return:.2%}")],
                 style={'width':'30%','display':'inline-block','textAlign':'center'}),
        html.Div([html.H3("Risk"), html.H2(f"{opt_risk:.2%}")],
                 style={'width':'30%','display':'inline-block','textAlign':'center'}),
        html.Div([html.H3("Sharpe Ratio"), html.H2(f"{sharpe:.2f}")],
                 style={'width':'30%','display':'inline-block','textAlign':'center'}),
    ]),

    html.Br(),

# ---- DROPDOWN ---- #
    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': i, 'value': i} for i in data.columns],
        value=data.columns[0],
        style={'color':'black'}
    ),

# ---- STOCK PERFORMANCE ---- #
    dcc.Graph(id='stock-graph'),

# ---- EFFICIENT FRONTIER ---- #
    dcc.Graph(
        figure={
            'data': [
                go.Scatter(
                    x=results[1,:],
                    y=results[0,:],
                    mode='markers',
                    marker=dict(
                        color=results[2,:],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Portfolios'
                ),
                go.Scatter(
                    x=[opt_risk],
                    y=[opt_return],
                    mode='markers',
                    marker=dict(color='red', size=14),
                    name='Optimal Portfolio'
                )
            ], 
            'layout': go.Layout(
                title='Efficient Frontier',
                xaxis={'title': 'Risk'},
                yaxis={'title': 'Return'},
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117'
             )
        }
    ),

# ---- PIE CHART ---- #
    dcc.Graph(
        figure={
            'data': [go.Pie(labels=stocks, values=weights)],
            'layout': go.Layout(title='Portfolio Allocation',
                                plot_bgcolor='#0d1117',
                                paper_bgcolor='#0d1117')
        }
    ),

# ---- CORRELATION HEATMAP ---- #
    dcc.Graph(
        figure=px.imshow(returns.corr(),
                         text_auto=True,
                         title="Correlation Heatmap")
    )

])

# ---------------- CALLBACK ---------------- #
@app.callback(
    Output('stock-graph', 'figure'),
    Input('stock-dropdown', 'value')
)
def update_graph(selected_stock):
    return {
        'data': [go.Scatter(x=data.index, y=data[selected_stock],
                            mode='lines', name=selected_stock)],
        'layout': go.Layout(title=f"{selected_stock} Price Trend",
                            plot_bgcolor='#0d1117',
                            paper_bgcolor='#0d1117')
    }

# ---------------- RUN ---------------- #
if __name__ == '__main__':
    app.run(debug=True)
