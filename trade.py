
import os, logging
from assets import *
import numpy as np
import pandas as pd
from scipy import stats
from datetime import date, timedelta
import alpaca_trade_api as tradeapi
from google.cloud import bigquery
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_APP_CRED

LOG = logging.getLogger("Algorithmic Trading")


class TradeManager(AssetManager):

    def __init__(self):
        super().__init__()
        self.clock = self.api.get_clock()
        LOG.info("Market is {}".format('open.' if self.clock.is_open else 'closed.'))

    def adj_portfolio(self, ptf_size):
        assets = self.get_active_assets()
        allocation = self.get_EF_allocation(portfolio_size=ptf_size, portfolio_value=10000)

    
    @property
    def market_hours(self, start_date=date.today()):
        """Function that returns market open and close hours."""
        start_date = start_date.strftime("%Y-%m-%d")
        calendar = self.api.get_calendar(start=start_date, end=start_date)[0]
        return calendar

    def get_existing_orders(self, status='all', **kwargs):
        """Function that returns list of existing orders"""
        return self.api.list_orders(status, kwargs)

    def place_order(self, tickr, qty, side, typ, tif='gtc', **kwargs):
        """
        Funtion that let's a user place an order by passing 
        appropriate options.

        Parameters
        ----------
        tickr {str} -- Tickr symbol of the desired stock 
                    e.g. AAPL, CAKE
        qty {int} -- Desired quantity
        side {str} -- Order type (sell/buy)
        typ {str} -- Execution type e.g. market, limit
        tif {str} -- Time in force e.g. gtc, opg
        **kwargs -- Keyword arguments for limit price, etc.
        """
        self.api.submit_order(tickr, qty, side, typ, tif, **kwargs)
    
    def get_prices_df(self, tickrs, start, end, num_bars, timeframe='day', **kwargs):
        # This should only run during weekdays
        # https://docs.alpaca.markets/api-documentation/web-api/market-data/bars/
        bars = self.api.get_barset(symbols=tickrs, start=start, end=end,
                                     timeframe=timeframe, limit=num_bars)
        df_bars = (bars.df.stack(level=0)
                    .reset_index()
                    .rename(columns={'level_1':'tickr', 'time':'date'}))
        df_bars['date'] = df_bars['date'].dt.strftime('%Y-%m-%d')
        return df_bars

    def get_EF_allocation(self, portfolio_size, portfolio_value):
        df_hist = query_quotes_history()
        df = self.__get_momentum(df_hist)
        date = df.date.max()
        df_top = df.loc[df['date'] == date]
        df_top = df_top.sort_values(by='momentum', ascending=False).head(portfolio_size)

        # Set the universe to the top momentum stocks for the period
        universe = df_top['tickr'].tolist()

        # Create a df with just the stocks from the universe
        df_u = df.loc[df['tickr'].isin(universe)]

        # Create the portfolio
        # Pivot to format for the optimization library
        df_u = df_u.pivot_table(
            index='date', 
            columns='tickr',
            values='close',
            aggfunc='sum'
            )

        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(df_u)
        S = risk_models.sample_cov(df_u)

        # Optimise the portfolio for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S, gamma=1) # Use regularization (gamma=1)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        # Allocate
        latest_prices = get_latest_prices(df_u)

        da = DiscreteAllocation(
            cleaned_weights,
            latest_prices,
            total_portfolio_value=portfolio_value
            )

        allocation = da.lp_portfolio()[0]
        return allocation

    def __get_momentum(self, df):
        df['momentum'] = (df.groupby('tickr')['close']
                            .rolling(MOMENTUM_WINDOW, min_periods=MINIMUM_MOMENTUM
                            ).apply(self.momentum_score, raw=False)
                            .reset_index(level=0, drop=True))
        return df

    @staticmethod
    def momentum_score(ts):
        x = np.arange(len(ts))
        log_ts = np.log(ts)
        regress = stats.linregress(x, log_ts)
        annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
        return annualized_slope * (regress[2] ** 2)


def store_quotes(df_quotes, table_name):
    # Store daily quotes to BigQuery
    client = bigquery.Client()
    table_id = f"{PROJECT}.{SCHEMA}.{table_name}"

    job_config = bigquery.LoadJobConfig(
    # Specifing schema
    schema=[
        bigquery.SchemaField("date", field_type=bigquery.enums.SqlTypeNames.STRING, 
        mode="REQUIRED", description="Extract date"),
        bigquery.SchemaField("tickr", field_type=bigquery.enums.SqlTypeNames.STRING, 
        mode="REQUIRED", description="Stock symbol/tickr"),
        bigquery.SchemaField("open", field_type=bigquery.enums.SqlTypeNames.FLOAT, 
        mode="REQUIRED", description="Opening price"),
        bigquery.SchemaField("close", field_type=bigquery.enums.SqlTypeNames.FLOAT, 
        mode="REQUIRED", description="Closing price"),
        bigquery.SchemaField("high", field_type=bigquery.enums.SqlTypeNames.FLOAT, 
        mode="REQUIRED", description="High price"),
        bigquery.SchemaField("low", field_type=bigquery.enums.SqlTypeNames.FLOAT, 
        mode="REQUIRED", description="Low price"),
    ],
    # Setting write disposition to replace existing data over appending
    write_disposition="WRITE_TRUNCATE",
    )

    job = client.load_table_from_dataframe(
        df_quotes, table_id, job_config=job_config
    )  # Make an API request.
    job.result()  # Wait for the job to complete.

    table = client.get_table(table_id)  # Make an API request.
    print(
        "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), table_id
        )
    )

def query_quotes_history():
    client = bigquery.Client()
    table_id = f"{PROJECT}.{SCHEMA}.quotes_history"
    sql = f"""
            select date, tickr, close
            from `{table_id}`
          """
    df = client.query(sql).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    return df


if __name__ == "__main__":
    tm = TradeManager()
    today = tm.market_hours.date
    since = today - timedelta(MOMENTUM_WINDOW)
    # print(tm.buying_power)
    # print(tm.get_active_assets('NASDAQ', 'NYSE'))
    # tm.place_order('CAKE', 1, 'buy', 'market', 'gtc')
    # print(tm.get_existing_orders())
    # df = tm.get_prices_df(tickrs=stockUniverse, timeframe='day', start=today, end=today)
    # store_quotes(df, table_name = "daily_quotes")
    # df_hist = tm.get_prices_df(tickrs=stockUniverse, timeframe='day', 
    #         start=since, end=today, num_bars=200)
    # store_quotes(df_hist, table_name = "quotes_history")
    # df = query_quotes_history()
    # tm.get_EF_allocation(portfolio_size=10, portfolio_value=10000)
    tm.adj_portfolio(ptf_size=10)


