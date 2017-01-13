import inspect

# https://stackoverflow.com/questions/31649936/set-class-attributes-from\
# -default-arguments#31650064
def auto_assign(func):
    signature = inspect.signature(func)

    def wrapper(*args, **kwargs):
        instance = args[0]
        bind = signature.bind(*args, **kwargs)
        for param in signature.parameters.values():
            if param.name != 'self':
                if param.name in bind.arguments:
                    setattr(instance, param.name, bind.arguments[param.name])
                if param.name not in bind.arguments and param.default is not param.empty:
                    setattr(instance, param.name, param.default)
        return func(*args, **kwargs)

    wrapper.__signature__ = signature # Restore the signature

    return wrapper
