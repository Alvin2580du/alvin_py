from .Car import SmallCar, LargeCar, MediumCar, PremiumCar
from .CarRentalSystem import CarRentalSystem
from .AdminSystem import AdminSystem
from .AuthenticationManager import AuthenticationManager
from .Customer import Customer, Admin


def bootstrap_system(auth_manager):

    admin_system = AdminSystem(auth_manager)
    system = CarRentalSystem(admin_system, auth_manager)

    rego = 0
    for name in ["Mazda", "Holden", "Ford"]:
        for model in ["Falcon", "Commodore", "Buggy"]:
            system.add_car(SmallCar(name, model, str(rego)))
            rego += 1
            system.add_car(MediumCar(name, model, str(rego)))
            rego += 1
            system.add_car(LargeCar(name, model, str(rego)))
            rego += 1

    for name in ["Tesla", "Audi", "Mercedes"]:
        for model in ["model x", "A4", "S class"]:
            system.add_car(PremiumCar(name, model, str(rego)))
            rego += 1

    for name in ["Matt", "Isaac", "Taylor"]:
        # Username, Password, Licence
        system.add_customer(Customer(name, 'pass', 1531))

    admin_system.add_admin(Admin('ian', '123'))

    return system
