def collect_data(type="training"):
    import datetime as dt
    import pandas_datareader.data as pdr

    date_entry_start = input('Enter start date in YYYY-MM-DD format: ')
    year, month, day = map(int, date_entry_start.split('-'))
    start = dt.date(year, month, day)

    date_entry_end = input('Enter end date in YYYY-MM-DD format: ')
    year, month, day = map(int, date_entry_end.split('-'))
    end = dt.date(year, month, day)

    ticker = input('Enter company ticker name: ')
    data = pdr.DataReader(ticker, 'yahoo', start, end)

    if type == 'training':
        data.to_csv('data/training.csv')
    elif type == 'testing':
        data.to_csv('data/testing.csv')

    return
