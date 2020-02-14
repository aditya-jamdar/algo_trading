from config import *
import logging
import pandas as pd
import alpaca_trade_api as tradeapi


class AssetManager():

    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

    def __init__(self):
        # Get account information
        self.account = self.api.get_account()
        self.buying_power = self.account.buying_power
        # Check if our account is restricted from trading
        if self.account.trading_blocked:
            raise Exception('Account is currently restricted from trading.')
    
    def get_positions(self):
        positions = self.api.list_positions()
        df = pd.DataFrame(
                {'ticker': [i.symbol for i in positions],
                'qty': [i.qty for i in positions],
                'market_value': [i.market_value for i in positions]}
            ).set_index('ticker')
        df['qty'] = df.qty.astype(int)
        return df

    def get_active_assets(self, *args):
        # Get active assests by filtering on exchange(s)
        active_assets = self.api.list_assets(status='active')
        if args:
            active_assets = [a for a in active_assets if a.exchange in args]
        return active_assets

    def __is_tradable(self, tickr):
        # Check if asset is tradable on the Alpaca platform
        asset = self.api.get_asset(tickr)
        return asset.tradable