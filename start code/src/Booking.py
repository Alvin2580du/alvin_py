from datetime import datetime
class Booking(object):

    def __init__(self, customer, start_date,end_date, car, location):
        self._customer  = customer
        ##self._period    = period
        self.start_date=start_date
        self.end_date=end_date
        self._car       = car
        self._location  = location

    @property
    def booking_fee(self):
        return self._car.get_fee(self.period)

    @property
    def location(self):
        return self._location

    @property
    def period(self):
        return (self.end_date-self.start_date).days+1

    def __str__(self):
        #-----------added start-------------
        date_format = "%Y-%m-%d"
        start_date=self.start_date.strftime(date_format)
        end_date=self.end_date.strftime(date_format)
        return f'Booking for:{self._customer},{self._car},period:{self.period} days,<{start_date},{end_date}>,{self.location}'
        #-----------added end-------------
        ##return f'Booking for: {self._customer}, {self._car}'
#-----------added start--------------
class BookingError(Exception):
    '''
    BookingError 
    '''
    def __init__(self,filed_name,message):
        Exception.__init__(self,message)
        self.filed_name=filed_name
        self.message=message

#-----------added end