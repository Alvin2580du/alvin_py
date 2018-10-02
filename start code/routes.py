from flask import render_template, request, redirect, url_for, abort
from flask_login import login_required, current_user

from server import app, system, auth_manager
from datetime import datetime
from src.Location import Location
from src.CarFactory import CarFactory
from src.Booking import BookingError


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Task 1: complete this function
    """
    if request.method == 'POST':
        #-----------added start-----------
        ##get username from request
        username=request.form['username']
        password=request.form['password']
        ##check username or passowrd is empty 
        if not (username and password):
            message="customer:username or password can not be empty !"
            return render_template('login.html',message=message) 

        ##login
        success=system.login_customer(username,password)

        ##invalid username or password return login page and show error message
        if not success:
            message="customer:invalid username or password !"
            return render_template('login.html',message=message) 
        #-----------added end--------------

        # Next helps with redirecting the user to their previous page
        redir = request.args.get('next')
        return redirect(redir or url_for('home'))
    
    return render_template('login.html')



@app.route('/login_admin', methods=['POST'])
def login_admin():

    '''
    TASK 4.1 complete this function
    '''
    #-----------added start-----------
    ##get username from request
    username=request.form['username']
    password=request.form['password']
    ##check username or passowrd is empty 
    if not (username and password):
        message="admin:username or password can not be empty !"
        return render_template('login.html',message=message) 

    success=system.login_admin(username,password)

    ##invalid username or password return login page and show error message
    if not success:
        message="admin:invalid username or password !"
        return render_template('login.html',message=message) 
    #-----------added end--------------
    redir = request.args.get('next')
    return redirect(redir or url_for('home'))



@app.route('/logout')
@login_required
def logout():
    auth_manager.logout()
    return redirect(url_for('home'))



'''
    Dedicated page for "page not found"
'''
@app.route('/404')
@app.errorhandler(404)
def page_not_found(e=None):
    return render_template('404.html'), 404



'''
    Search for Cars
'''
@app.route('/cars')
@login_required
def cars():
    """
    Task 2: At the moment this endpoint does not do anything if a search
    is sent. It should filter the cars depending on the search criteria
    """
    #-----------added start-------------
    #get keyword from url 
    name=request.args.get("make")
    model=request.args.get("model")
    if current_user.is_admin():
        cars=system.cars
    else:
        cars=system.car_search(name,model)

    ##cars = system.cars
    #----------added end--------------
    return render_template('cars.html', cars = cars)



'''
    Display car details for the car with given rego
'''
@app.route('/cars/<rego>')
@login_required
def car(rego):
    car = system.get_car(rego)

    if not car:
        abort(404)

    return render_template('car_details.html', car=car)



'''
    Make a booking for a car with given rego
'''
@app.route('/cars/book/<rego>', methods=["GET", "POST"])
@auth_manager.customer_required
def book(rego):
    car = system.get_car(rego)

    if not car:
        abort(404)
    
    if request.method == 'POST':
        #---------added start----------
      
        ## date_format = "%Y-%m-%d"
        ## start_date  = datetime.strptime(request.form['start_date'], date_format)
        ## end_date    = datetime.strptime(request.form['end_date'],   date_format)
        ##num_days = (end_date - start_date).days + 1


        if 'check' in request.form:
            ##fee = car.get_fee(num_days)
            try:
                fee=system.check_fee(car, request.form['start_date'], request.form['end_date'])
            
                return render_template(
                    'booking_form.html',
                    confirmation=True,
                    form=request.form,
                    car=car,
                    fee=fee
                    )

            except BookingError as e:
                return render_template('booking_form.html', error_message=e.message)
        elif 'confirm' in request.form:
            #-------------added start---------------
            ##location = Location(request.form['start'], request.form['end'])
            ##booking  = system.make_booking(current_user, num_days, car, location)
            try:
                booking  = system.make_booking(current_user, request.form['start_date'],
                                    request.form['end_date'], car, request.form['start'],request.form['end'])
                return render_template('booking_confirm.html', booking=booking)
                

            except BookingError as e:
                return render_template('booking_form.html', error_message=e.message)
           #-------------added end-------------

    return render_template('booking_form.html', car=car)


'''
    Display list of all bookings for the car with given 'rego'
'''
@app.route('/cars/bookings/<rego>')
@login_required
def car_bookings(rego):
    """
    Task 3: This should render a new template that shows a list of all
    the bookings associated with the car represented by 'rego'
    """
    #----------added start------------
    car=system.get_car(rego)
    return render_template('bookings.html',bookings=car.bookings)
    #----------added end---------------

    ##return render_template('bookings.html')


'''
    View all current bookings
'''
@app.route('/cars/bookings')
@auth_manager.admin_required
def all_bookings():

    '''
    Task 4.2: This should render a list of all current bookings
    on the system
    '''
    #-------------added start------------
    bookings=[]
    for car in system.cars:
        bookings.extend(car.bookings)
    return render_template('bookings.html',bookings=bookings)
    #------------added end-------------
    ##return '<h1> Needs to be implemented </h1>'


'''
    Add a new car to the system
'''
from src.CarFactory import CarFactory
@app.route('/add_car', methods=['GET', 'POST'])
@auth_manager.admin_required
def add_car():

    '''
    Task 4.3: This should allow the admin to register
    new cars to the system, using the method provided
    in the CarFactory class.
    
    Provide meaningful message upon success
    '''
    #--------added start----------

    if request.method == 'POST':

        name=request.form['name']
        model=request.form['model']
        rego=request.form['rego']
        car_type=request.form['car-type']
        filed_names=['name','model','rego','car type']
        error_message=[]
        ##the filed value cannot be empty
        for filed,filed_name in zip([name,model,rego,car_type],filed_names):
            print()
            f"filed:{filed},filedname:{filed_name}"
            if not filed:
                error_message.append(filed_name +" cannot be empty!")
        if  error_message :
            return render_template('car_register_form.html',error_message=error_message)

        car_factory=CarFactory()
        car=car_factory.make_car(name,model,rego,car_type)
        system.add_car(car)
        return render_template('car_register_form.html',register_success=True,car=car)
    #--------added end-------------
    return render_template('car_register_form.html')