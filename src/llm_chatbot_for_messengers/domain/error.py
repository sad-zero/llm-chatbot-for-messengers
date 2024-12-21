class SpecificationError(RuntimeError):
    pass


class ParserError(RuntimeError):
    pass


class ResourceError(RuntimeError):
    pass


class DataError(RuntimeError):
    pass


class WorkflowError(RuntimeError):
    pass

class FactoryError(RuntimeError):
    pass