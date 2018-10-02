from flask_login import UserMixin
from abc import ABC, abstractmethod


class User(UserMixin, ABC):
    __id = -1

    def __init__(self, username, password):
        self._id = self._generate_id()
        self._username = username
        self._password = password

    @property
    def username(self):
        return self._username

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        """Required by Flask-login"""
        return str(self._id)

    def _generate_id(self):
        User.__id += 1
        return User.__id

    def validate_password(self, password):
        return self._password == password

    @abstractmethod
    def is_admin(self):
        pass


class Customer(User):

    def __init__(self, username, password, licence):
        super().__init__(username, password)
        self._licence = licence

    def is_admin(self):
        return False

    def __str__(self):
        return f'Customer <name: {self._username}, licence: {self._licence}>'


class Admin(User):

    def is_admin(self):
        return True

    def __str__(self):
        return f'Admin <name: {self._username}>'
