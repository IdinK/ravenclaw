from collections import OrderedDict
import pandas as pd


lotr = pd.DataFrame(OrderedDict({
    'number': range(10),
    'firstname': [
        'Frodo', 'Aragorn', 'Gandalf', 'Legolas', 'Saruman', 
        'Bilbo', 'Gollum', 'Arwen', 'Galadriel', 'Boromir'],
    'lastname': [
        'Baggins', None, None, None, None,
        'Baggins', None, None, None, None
    ],
    'date': pd.Series(pd.date_range(start = pd.datetime.today(), periods = 10)).dt.date
}))

