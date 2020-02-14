
import os, logging
import numpy as np
from assets import *
from scipy import stats
from datetime import date, timedelta
from google.cloud import bigquery
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_APP_CRED

LOG = logging.getLogger("Algorithmic Trading")

    
class TradeManager(AssetManager):
    """Class to query quotes data and perform portfolio adjustment."""

    def __init__(self):
        super().__init__()
        self.clock = self.api.get_clock()
        LOG.info("Market is {}".format('open.' if self.clock.is_open else 'closed.'))

    def adj_portfolio(self, ptf_size):
        """
        Method that places buy/sell orders based on optimized allocation.

        Parameters
        ----------
        ptf_size {int} -- Desired portfolio size
        """
        positions = self.get_positions()
        allocation = self.__get_ef_allocation(portfolio_size=ptf_size, portfolio_value=10000)
        # Collect sell orders
        df = positions.merge(allocation, how='left', left_index=True, right_index=True).fillna(0)
        df_sell = (df.assign(sell_qty=np.where(df.qty>df.allocation, df.qty-df.allocation, 0), side='sell')
                        .drop(columns=['qty', 'allocation', 'market_value'])
                        .rename(columns={'sell_qty':'qty'}).reset_index())
  
        # Collect buy orders
        df = (positions.merge(allocation, how='right', left_index=True, right_index=True).fillna(0)
                    .reset_index().rename(columns={'index':'ticker'}))
        df_buy = (df.assign(buy_qty=np.where(df.qty<df.allocation, df.allocation-df.qty, 0), side='buy')
                        .drop(columns=['qty', 'allocation', 'market_value'])
                        .rename(columns={'buy_qty':'qty'}))
        
        # Place buy/sell orders
        pd.concat([df_sell, df_buy]).query("qty>0").apply(
            lambda x: self.__place_order(x.ticker, x.qty, x.side, typ='market'), axis=1)
    
    @property
    def market_hours(self, start_date=date.today()):
        """Property that returns market open and close hours."""
        start_date = start_date.strftime("%Y-%m-%d")
        calendar = self.api.get_calendar(start=start_date, end=start_date)[0]
        return calendar

    def __get_existing_orders(self, status='all', **kwargs):
        """Method that returns list of existing orders"""
        return self.api.list_orders(status, kwargs)

    def __place_order(self, tickr, qty, side, typ, tif='gtc', **kwargs):
        """
        Method that let's a user place an order by passing 
        appropriate options.

        Parameters
        ----------
        tickr {str} -- Tickr symbol of the desired stock 
                        e.g. AAPL, CAKE
        qty {int}   -- Desired quantity
        side {str}  -- Order type (sell/buy)
        typ {str}   -- Execution type e.g. market, limit
        tif {str}   -- Time in force e.g. gtc, opg
        **kwargs    -- Keyword arguments for limit price, etc.
        """
        LOG.info(f"{side} {qty} of {tickr}")
        self.api.submit_order(tickr, qty, side, typ, tif, **kwargs)
    
    def get_prices_df(self, tickrs, start, end, num_bars, timeframe='day', **kwargs):
        """
        Method that queries prices using Alpaca paper trading api.

        Parameters
        ----------
        tickr {str}      -- Tickr symbol of the desired stock 
                            e.g. AAPL, CAKE
        start {DateTime} -- From date
        end {DateTime}   -- Till date
        num_bars {int}   -- Maximum no. of bars desired
        timeframe {str}  -- Desired time frame 
        **kwargs         -- Keyword arguments for limit price, etc.

        Returns
        -------
        pd.DataFrame -- dataframe with prices
        """
        # This should only run during weekdays
        bars = self.api.get_barset(symbols=tickrs, start=start, end=end,
                                     timeframe=timeframe, limit=num_bars)
        df_bars = (bars.df.stack(level=0)
                    .reset_index()
                    .rename(columns={'level_1':'tickr', 'time':'date'}))
        df_bars['date'] = df_bars['date'].dt.strftime('%Y-%m-%d')
        return df_bars

    def __get_ef_allocation(self, portfolio_size, portfolio_value):
        """Method that returns optimal allocation based on Markowitz Portfolio theory."""
        df_hist = query_quotes_history()
        df = self.__get_momentum(df_hist)
        date = df.date.max()
        df_top = df.loc[df['date'] == date]
        df_top = df_top.sort_values(by='momentum', ascending=False).head(portfolio_size)

        universe = df_top['tickr'].tolist()
        df_u = df.loc[df['tickr'].isin(universe)]

        df_u = df_u.pivot_table(
            index='date', 
            columns='tickr',
            values='close',
            aggfunc='sum'
            )

        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(df_u)
        S = risk_models.sample_cov(df_u)

        # Optimise the portfolio for maximal Sharpe ratio 
        # with regularization
        ef = EfficientFrontier(mu, S, gamma=1)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        latest_prices = get_latest_prices(df_u)

        # Generate allocation
        da = DiscreteAllocation(
            cleaned_weights,
            latest_prices,
            total_portfolio_value=portfolio_value
            )

        allocations = pd.Series(da.lp_portfolio()[0], name='allocation')
        return allocations

    def __get_momentum(self, df):
        df['momentum'] = (df.groupby('tickr')['close']
                            .rolling(MOMENTUM_WINDOW, min_periods=MINIMUM_MOMENTUM
                            ).apply(self.__momentum_score, raw=False)
                            .reset_index(level=0, drop=True))
        return df

    @staticmethod
    def __momentum_score(ts):
        x = np.arange(len(ts))
        log_ts = np.log(ts)
        regress = stats.linregress(x, log_ts)
        annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
        return annualized_slope * (regress[2] ** 2)


def store_quotes(df_quotes, table_name):
    """Funtion to store daily quotes to BigQuery."""
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
    """Helper function to query quotes stored in BigQuery."""
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
    # Instantiate the trading class and begin portfolio adjustment
    LOG.info("Starting portfolio adjustment...")
    tm = TradeManager()
    today = tm.market_hours.date
    since = today - timedelta(MOMENTUM_WINDOW)

    # Query quotes for today
    latest_quotes = tm.get_prices_df(tickrs=stockUniverse, timeframe='day', 
            start=today, end=today, num_bars=200)
    store_quotes(latest_quotes, table_name = "daily_quotes")
    LOG.info("Today's quotes stored to BQ")

    # Update historical quotes and store to BQ
    quotes_hist = tm.get_prices_df(tickrs=stockUniverse, timeframe='day', 
            start=since, end=today, num_bars=200)
    store_quotes(quotes_hist, table_name = "quotes_history")
    LOG.info("Historical daily quotes updated in BQ")

    # Perform adjustment
    tm.adj_portfolio(ptf_size=10)
    LOG.info("Portfolio adjustment is complete.")


