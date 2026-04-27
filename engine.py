class Value:
    "stores the scalar value and its gradient"
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda:    None
        self._prev = (_children)
        self._op = _op 
   