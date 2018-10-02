

class AdminSystem():

    def __init__(self, auth_manager):
        self._auth_manager = auth_manager
        self._admins = []

    def add_admin(self, admin):
        self._admins.append(admin)

    def get_user_by_id(self, user_id):
        for admin in self._admins:
            if admin.get_id() == user_id:
                return admin
                
        return None
            
    def login(self, username, password):
        for admin in self._admins:
            if self._auth_manager.login(admin, username, password):
                return True
        return False










