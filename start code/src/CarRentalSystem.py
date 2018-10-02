from src.Booking import Booking, BookingError
from src.Location import Location
import copy
from datetime import datetime


class CarRentalSystem:
    def __init__(self, admin_system, auth_manager):
        self._cars = []
        self._customers = []
        self._bookings = []
        self._admin_system = admin_system
        self._auth_manager = auth_manager

    '''
    Query Processing Services
    '''

    def car_search(self, name=None, model=None):
        # ---------added start-----------
        cars = []
        for car in self.cars:
            ##if name not is None and not empty string ,then it is a word for search
            # so,if name not euqal car.name ,look at next car
            if not name is None and name != "" and car.name != name:
                continue
            ##the same as  above
            if not model is None and model != "" and car.model != model:
                continue
            ##only show available car for customer
            ## if not car.is_available :
            ##     continue
            cars.append(car)
        return cars
        # ---------added end--------

        # return []

    def get_user_by_id(self, user_id):
        for c in self._customers:
            if c.get_id() == user_id:
                return c

        return self._admin_system.get_user_by_id(user_id)

    def get_car(self, rego):
        for c in self.cars:
            if c.rego == rego:
                return c
        return None

    '''
    Booking Services
    '''

    # def make_booking(self, customer, period, car, location):
    #     # Prevent the customer from referencing 'current_user';
    #     # otherwise the customer recorded in each booking will be modified to
    #     # a different user whenever the current_user changes (i.e. when new user logs-in)
    #     customer = copy.copy(customer)

    #     new_booking = Booking(customer, period, car, location)
    #     self._bookings.append(new_booking)
    #     car.add_booking(new_booking)
    #     return new_booking

    # ----------------added start------------
    def make_booking(self, customer, input_start_date, input_end_date, car, start_location, end_location):
        # Prevent the customer from referencing 'current_user';
        customer = copy.copy(customer)
        ##check fileds are not empty
        if not start_location:
            raise BookingError("start_location", "Specify a valid start loaction!")
        if not end_location:
            raise BookingError("end_location", "Specify a valid end loaction!")

        if not input_start_date:
            raise BookingError("start_date", "Specify a valid start date!")
        if not input_end_date:
            raise BookingError("end_date", "Specify a valid end date!")
            ##check end

        ##check end_date < start_date 
        date_format = "%Y-%m-%d"
        start_date = datetime.strptime(input_start_date, date_format)
        end_date = datetime.strptime(input_end_date, date_format)
        if end_date < start_date:
            raise BookingError("period", "Specify a valid booking period!")
            ##check end

        ##all filed valid ,then make booking
        location = Location(start_location, end_location)
        new_booking = Booking(customer, start_date, end_date, car, location)
        self._bookings.append(new_booking)
        car.add_booking(new_booking)
        return new_booking

    def check_fee(self, car, input_start_date, input_end_date):

        ##check fileds are not empty
        if not input_start_date:
            raise BookingError("start_date", "Specify a valid start date!")
        if not input_end_date:
            raise BookingError("end_date", "Specify a valid end date!")

            ##check end_date < start_date
        date_format = "%Y-%m-%d"
        start_date = datetime.strptime(input_start_date, date_format)
        end_date = datetime.strptime(input_end_date, date_format)
        if end_date < start_date:
            raise BookingError("period", "Specify a valid booking period!")
            ##check end

        ##all fileds are valid,then compute fee
        num_days = (end_date - start_date).days + 1
        return car.get_fee(num_days)


        # ---------------added end--------------

    '''
    Registration Services
    '''

    def add_car(self, car):
        self._cars.append(car)

    def add_customer(self, customer):
        self._customers.append(customer)

    '''
    Login Services
    '''

    def login_customer(self, username, password):
        for customer in self._customers:
            if self._auth_manager.login(customer, username, password):
                return True
        return False

    def login_admin(self, username, password):
        return self._admin_system.login(username, password)

    '''
    Properties
    '''

    @property
    def cars(self):
        return self._cars

    @property
    def bookings(self):
        return self._bookings
