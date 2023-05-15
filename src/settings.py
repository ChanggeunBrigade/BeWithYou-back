from dotenv import dotenv_values

config = dotenv_values(".env")


class SingletonMeta(type):
    _instances = {}

    def __new__(cls, name, bases, attrs):
        if name in cls._instances:
            return cls._instances[name]
        else:
            instance = super().__new__(cls, name, bases, attrs)
            cls._instances[name] = instance
            return instance
