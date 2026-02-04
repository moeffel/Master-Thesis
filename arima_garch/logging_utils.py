# Logging utilities
import logging

log = logging.getLogger("arima_garch")


class CoinLogFilter(logging.Filter):
    def __init__(self, coin_id="N/A"):
        super().__init__()
        self.coin_id = coin_id

    def filter(self, record):
        record.coin_id = self.coin_id
        return True


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-12s - %(levelname)-8s - %(funcName)-25s [%(coin_id)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    _log = logging.getLogger("arima_garch")
    _filter = CoinLogFilter()
    _log.addFilter(_filter)
    return _log, _filter
