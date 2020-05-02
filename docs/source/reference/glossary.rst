========
Glossary
========

.. glossary::

compound structure
    Any structure that cannot be broken down into smaller, simpler parts.
    A subcircuit is an example of a complex structure; it contains simpler
    elements (or other compound structures) connected internally to form 
    the overall larger structure.

decorator
    An operator that transforms a function.  For example, a ``log``
    decorator may be defined to print debugging information upon
    function execution::

        >>> def log(f):
        ...     def new_logging_func(*args, **kwargs):
        ...         print("Logging call with parameters:", args, kwargs)
        ...         return f(*args, **kwargs)
        ...
        ...     return new_logging_func

    Now, when we define a function, we can "decorate" it using ``log``::

        >>> @log
        ... def add(a, b):
        ...     return a + b

    Calling ``add`` then yields:

    >>> add(1, 2)
    Logging call with parameters: (1, 2) {}
    3

dictionary
    Resembling a language dictionary, which provides a mapping between
    words and descriptions thereof, a Python dictionary is a mapping
    between two objects::

        >>> x = {1: 'one', 'two': [1, 2]}

    Here, `x` is a dictionary mapping keys to values, in this case
    the integer 1 to the string "one", and the string "two" to
    the list ``[1, 2]``.  The values may be accessed using their
    corresponding keys::

        >>> x[1]
        'one'

        >>> x['two']
        [1, 2]

    Note that dictionaries are not stored in any specific order.  Also,
    most mutable (see *immutable* below) objects, such as lists, may not
    be used as keys.

    For more information on dictionaries, read the
    `Python tutorial <https://docs.python.org/tutorial/>`_.

instance
    A class definition gives the blueprint for constructing an object::

        >>> class House:
        ...     wall_colour = 'white'

    Yet, we have to *build* a house before it exists::

        >>> h = House() # build a house

    Now, ``h`` is called a ``House`` instance.  An instance is therefore
    a specific realisation of a class.

iterable
    A sequence that allows "walking" (iterating) over items, typically
    using a loop such as::

        >>> x = [1, 2, 3]
        >>> [item**2 for item in x]
        [1, 4, 9]

    It is often used in combination with ``enumerate``::
        >>> keys = ['a','b','c']
        >>> for n, k in enumerate(keys):
        ...     print("Key %d: %s" % (n, k))
        ...
        Key 0: a
        Key 1: b
        Key 2: c

list
    A Python container that can hold any number of objects or items.
    The items do not have to be of the same type, and can even be
    lists themselves::

        >>> x = [2, 2.0, "two", [2, 2.0]]

    The list `x` contains 4 items, each which can be accessed individually::

        >>> x[2] # the string 'two'
        'two'

        >>> x[3] # a list, containing an integer 2 and a float 2.0
        [2, 2.0]

    It is also possible to select more than one item at a time,
    using *slicing*::

        >>> x[0:2] # or, equivalently, x[:2]
        [2, 2.0]

    In code, arrays are often conveniently expressed as nested lists::


        >>> np.array([[1, 2], [3, 4]])
        array([[1, 2],
            [3, 4]])

    For more information, read the section on lists in the `Python
    tutorial <https://docs.python.org/tutorial/>`_.  For a mapping
    type (key-value), see *dictionary*.

method
    A function associated with an object.  For example, each ndarray has a
    method called ``repeat``::

        >>> x = np.array([1, 2, 3])
        >>> x.repeat(2)
        array([1, 1, 2, 2, 3, 3])

reference
    If ``a`` is a reference to ``b``, then ``(a is b) == True``.  Therefore,
    ``a`` and ``b`` are different names for the same Python object.

self
    Often seen in method signatures, ``self`` refers to the instance
    of the associated class.  For example:

        >>> class Paintbrush:
        ...     color = 'blue'
        ...
        ...     def paint(self):
        ...         print("Painting the city %s!" % self.color)
        ...
        >>> p = Paintbrush()
        >>> p.color = 'red'
        >>> p.paint() # self refers to 'p'
        Painting the city red!

tuple
    A sequence that may contain a variable number of types of any
    kind.  A tuple is immutable, i.e., once constructed it cannot be
    changed.  Similar to a list, it can be indexed and sliced::

        >>> x = (1, 'one', [1, 2])
        >>> x
        (1, 'one', [1, 2])

        >>> x[0]
        1

        >>> x[:2]
        (1, 'one')

    A useful concept is "tuple unpacking", which allows variables to
    be assigned to the contents of a tuple::

        >>> x, y = (1, 2)
        >>> x, y = 1, 2

    This is often used when a function returns multiple values:

        >>> def return_many():
        ...     return 1, 'alpha', None

        >>> a, b, c = return_many()
        >>> a, b, c
        (1, 'alpha', None)

        >>> a
        1
        >>> b
        'alpha'
        