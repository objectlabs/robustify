import collections

import sys
import types
import inspect

import operator
import logging

from time import sleep, time as now


###############################################################################
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############################################################################
# Dealing with explosions - the Context Manager way
###############################################################################
def robust_execution_context(handle_it=None, abort_phrase="aborting",
                             logger=None,
                             resilient_rollback=False):
    """
    Create a context to surround execution of a multi-step, failure-prone
    process.

    :param handle_it: thunk that gets executed first post-kaboom
    :param str abort_phrase: description of what's happening post-kaboom
    :param logger: probably pointless, but ... you can pass yours.
    :param bool resilient_rollback: keep executing rollback steps even if THEY
    kaboom
    :return: a Context Manager for controlled explosion management
    """
    return RobustExecutionContext(handle_it, abort_phrase, logger,
                                  resilient_rollback)

###############################################################################
class RobustExecutionContext(object):

    def __init__(self, handle_it=None, abort_phrase="aborting", logger=None,
                 resilient_rollback=False):
        self._handle_it = (lambda: 42) if handle_it is None else handle_it
        self._logger = logger_for_sure(logger)
        self._resilient_rollback = resilient_rollback
        self._abort_phrase = abort_phrase
        self._abort_banner = _banner("WHOOPS!", abort_phrase,
                                     "& rolling back!")
        # IDE mollification ==> mention these here in constructor:
        self._rollback_hook = self._cleanup_hook = \
            self._begin_to = self._abort = None

    ###########################################################################
    def __enter__(self):
        def handle_it_n_rollback():
            self._handle_it()
            self.rollback()
        #set things up
        self._rollback_hook = Hook(lifo=True)
        self._cleanup_hook = Hook()
        (self._begin_to, self._abort) = make_abort_handler(
            handle_it_n_rollback,
            self._abort_banner,
            self._logger,
            swallow=True)

        return self

    ###########################################################################
    def __exit__(self, exc_type, exc_value, exc_traceback):
        #tear things down
        if exc_type is None:
            self.cleanup()
        else:
            self.abort(exc_type, exc_value, exc_traceback, swallow=True)
            return False        # i.e., will propagate error/exception

    ###########################################################################
    def _resilientify(self, f):
        if self._resilient_rollback:
            oopsfmt = "Ignoring problem during rollback step: %s"
            logoops = lambda e: self._logger.error(oopsfmt % e)
            return robustify(max_attempts=1, do_on_exception=logoops)(f)
        else:
            return f

    ###########################################################################
    def push_rollback_step(self, f):
        self._rollback_hook.add(self._resilientify(f))

    ###########################################################################
    def clear_rollback_steps(self):
        self._rollback_hook.clear()

    ###########################################################################
    def push_cleanup_step(self, f):
        self._cleanup_hook.add(f)

    ###########################################################################
    def push_unwind_step(self, f):
        """Pushes f so as to execute during either normal OR abortive exit."""
        self.push_rollback_step(f)
        self.push_cleanup_step(f)
    ###########################################################################
    def begin_to(self, phase, morespew=""):
        self._begin_to(phase, morespew)

    ###########################################################################
    # noinspection PyUnusedLocal
    def abort(self, exc_type, exc_value, exc_traceback, swallow=False):
        self._abort(exc_value, swallow=swallow)

    ###########################################################################
    def rollback(self):
        self._rollback_hook.run()
        self.clear_rollback_steps()

    ###########################################################################
    def cleanup(self):
        self._cleanup_hook.run()

###############################################################################
def _banner(*msg_parts):
    banner_bracket = " *** !!! *** !!! *** !!! *** "
    return banner_bracket + " ".join(msg_parts) + banner_bracket


###############################################################################
# Dealing with explosions - the Python 3 ripoff way
###############################################################################


class NestedContextsMegaManager(object):
    # in Python 3, you'd use an ExitStack for this shit.
    # for more context, see http://stackoverflow.com/questions/16083791
    # (get it?  CONTEXT?  haaaaaahahaha.)

    ###########################################################################
    def __init__(self, underling_managers):
        self.children = underling_managers

    ###########################################################################
    def __enter__(self):
        for context in self.children:
            context.__enter__()

    ###########################################################################
    def __exit__(self, exc_type, exc_val, exc_tb):
        kaboom_du_now = (exc_type, exc_val, exc_tb)
        while self.children:
            # noinspection PyBroadException
            try:
                context = self.children.pop()
                suppress = context.__exit__(*kaboom_du_now)
                if suppress:
                    kaboom_du_now = (None, None, None)
            except:
                kaboom_du_now = sys.exc_info()
                if not self.children:
                    raise
        return False


###############################################################################
# Dealing with explosions - the Functional way
###############################################################################

def reraise(new_exc):
    (et, ev, etb) = sys.exc_info()
    raise new_exc, None, etb

###############################################################################
def make_abort_handler(handle_it, abort_phrase="aborting", logger=None,
                       swallow=False):
    _phase = { 'ima' : "get it done" }  # so you call this a closure...
    logger = logger_for_sure(logger)

    def begin_to(phase, morespew=""):
        _phase['ima'] = phase
        logger.info("Setting about to %s %s..." % (phase, morespew))

    def abort(err=None, swallow=swallow):
        (et, ev, etb) = sys.exc_info()
        whats_happening = ("%s after failing to %s" %
                           (abort_phrase, _phase['ima']))
        logger.error("%s : %s" % (whats_happening, err or "Ow."))
        logger.info(_banner(" handler execution commencing "))
        try:
            handle_it()
        except Exception, e:
            logger.error("Exception while %s : %s" % (whats_happening, e))
            reraise(e)
        except:
            logger.error("Unspecified error while %s - gosh!" %
                         whats_happening)
            raise
        finally:
            logger.info(_banner(" handler execution concluded "))

        # Successfully handled original issue; now resume aborting
        if err is None:
            if not swallow:
                raise
        else:
            if not swallow:
                raise err, None, etb
    return (begin_to, abort)

###############################################################################
class Hook(object):
    """
    A Hook is a place to hang sequences of functions you'd like to run
    potentially at some future time.
    Order is guaranteed; `lifo` arg to constructor says it's reversed.
    """
    def __init__(self, lifo=False):
        self.hook_functions = []
        self.lifo = lifo

    def clear(self):
        self.hook_functions = []

    def add(self, hook_fcn):
        if hook_fcn:
            self.hook_functions.append(hook_fcn)

    def run(self):
        stuff_to_do = (self.hook_functions[::-1] if self.lifo else
                       self.hook_functions)
        return [hfcn() for hfcn in stuff_to_do]


###############################################################################
def logger_for_sure(a_logger_maybe):
    if a_logger_maybe is not None:
        almbrs = dict(inspect.getmembers(a_logger_maybe))
        if all(map(lambda n: (n in almbrs) and inspect.isroutine(almbrs[n]),
                   ['error', 'warn', 'info', 'debug'])):
            return a_logger_maybe
    return NoOpLogger()

###############################################################################
class NoOpLogger:
    def error(*_, **__):
        pass

    def warn(*_, **__):
        pass

    def info(*_, **__):
        pass

    def debug(*_, **__):
        pass

###############################################################################
def die_with_err(death_rattle):
    """
    Raise an Exception based on the supplied death_rattle, which may be
    (a) an Exception, in which case it is raised;
    (b) a value that may be supplied to the Exception() constructor,
        the result of which is then raised; or,
    (c) a thunk returning a value or an Exception, which result is
        then treated as in (b) or (a), respectively.
    """
    not_an_exception = raise_or_return(do_if_doable(death_rattle) or
                                       death_rattle)
    raise Exception(not_an_exception)


###############################################################################
# Living with properties (are they attributes? items? "pythonic"?)
###############################################################################

def getprop(x, prop_name, val_if_missing=None):
    """
    Returns the given property of its first argument.
    """
    x_has_prop = x and hasattr(x, "has_key") and (prop_name in x)
    return x[prop_name] if x_has_prop else val_if_missing

###############################################################################
def getprop_only_more_so(x, prop_name, val_if_missing=None):
    """
    Return the given property of its first argument.
    By 'property' we mean 'item' ... or, failing that, 'attr'.
    CAREFUL WITH THIS; it's subtly different from the above,
    and not always in a good way.
    """
    if x:
        if hasattr(x, "has_key") and (prop_name in x):
            return x[prop_name]
        if hasattr(x, prop_name):
            return getattr(x, prop_name)
    return val_if_missing

###############################################################################
def getprop_dwim(x, prop_spec, **kwargs):
    return getprop_sequence(x, prop_spec_parts(prop_spec), **kwargs)

###############################################################################
def getprop_or_die(x, prop_spec, die_with=None):
    sentinel_obj = object()
    result = getprop_dwim(x, prop_spec, val_if_missing=sentinel_obj)
    if result is sentinel_obj:
        die_with_err(die_with or "Cannot get value for '%s'" % prop_spec)
    else:
        return result

###############################################################################
def prop_spec_parts(prop_spec):
    """ Return the canonical 'sequence' representation of prop_spec
        (aka list of atomic field names, like ["foo", "bar"])
    """
    if isinstance(prop_spec, basestring):  # e.g., "foo.bar.baz"
        assert isinstance(prop_spec, str) or isinstance(prop_spec, unicode)
        # so now the IDE won't carp about this:
        return prop_spec.split('.')
    elif isinstance(prop_spec, list):      # e.g., ["foo", "bar", "baz"]
        return prop_specs_join(*prop_spec)
    else:                                  # e.g., 3
        return [ prop_spec ]

###############################################################################
def prop_specs_join(*prop_spec_pieces):
    """
    Return the canonical 'sequence" representation obtained by
    concatenating all the supplied partial prop_specs together.

        >>> prop_specs_join("a.b", 3, ["d", 4, "e.f"])
        ['a', 'b', 3, 'd', 4, 'e', 'f']

    """
    result = []
    for part in map(prop_spec_parts, prop_spec_pieces):
        result += part
    return result

###############################################################################
def safe_getprop(x, prop_name, val_if_missing=None, val_if_error=None):
    # noinspection PyBroadException
    try:
        return getprop(x, prop_name, val_if_missing)
    except Exception:
        return val_if_missing if val_if_error is None else val_if_error

###############################################################################
def get_dict_prop(x, dict_name, prop_name_within_dict, val_if_missing=None):
    """
    If dict_name is a property of x whose value is a dict,
    returns the indicated property value from within that dict.
    Otherwise, val_if_missing.
    """
    the_dict = getprop(x, dict_name)
    return getprop(the_dict, prop_name_within_dict, val_if_missing)

###############################################################################
def getprop_chain(orig_x, *prop_seq, **kwargs):
    """
    Follow the given chain of properties, starting from orig_x.
    Example:

        >>> x = { 'a' : { 'b': { 'c' : 'surprise!' } } }
        >>> getprop_chain(x, 'a', 'b')
        {'c': 'surprise!'}

    Optional keyword arguments:
    :param val_if_missing: value to return if one of the properties is missing
    :param safe: swallow exceptions -- e.g.,
                        >>> getprop_chain(6, 'a', safe=True) # ==> None

    :param short_circuit: if True, upon encountering the first missing
                            property,
                          immediately return the val_if_missing value.
                          if False, try to retrieve
                          the next property from that value -- e.g.,
                        >>> getprop_chain(x, 'a', 'c', 'd', 'c',
                        ...               val_if_missing={ 'd': { 'c': 5 } })
                        {'d': {'c': 5}}
                        >>> getprop_chain(x, 'a', 'c', 'd', 'c',
                        ...               val_if_missing={ 'd': { 'c': 5 } },
                        ...               short_circuit=False)
                        5
    """
    return getprop_sequence(orig_x, prop_seq, **kwargs)

###############################################################################
def getprop_sequence(orig_x, prop_seq,
                     val_if_missing=None, safe=False, short_circuit=True):
    """
    Exactly like getprop_chain(), but property chain is a single list argument.
    Exactly the same examples:

        >>> x = { 'a' : { 'b': { 'c' : 'surprise!' } } }
        >>> getprop_sequence(x, ['a', 'b'])
        {'c': 'surprise!'}
        >>> ac_dc = ['a', 'c', 'd', 'c']
        >>> getprop_sequence(x, ac_dc)
        >>> getprop_sequence(x, ac_dc, val_if_missing={ 'd': { 'c': 5 } })
        {'d': {'c': 5}}
        >>> getprop_sequence(x, ac_dc, val_if_missing={ 'd': { 'c': 5 } },
        ...                  short_circuit=False)
        5
    """
    x = orig_x
    der_proppengetten = safe_getprop if safe else getprop
    __SENTINEL__ = object()     # guaranteed unique!
    for prop in prop_seq:
        x = der_proppengetten(x, prop, val_if_missing=__SENTINEL__)
        if x == __SENTINEL__ :
            x = val_if_missing
            if short_circuit:
                break
    return x

###############################################################################
def getprop_star(orig_x, prop_path,
                 val_if_missing=None, safe=False, short_circuit=True):
    """
    Follow the "property path", starting from orig_x.
    Example:

        >>> x = { 'a' : { 'b': { 'c' : 'surprise!' } } }
        >>> getprop_star(x, 'a.b')
        {'c': 'surprise!'}

    """
    return getprop_sequence(orig_x, prop_path.split('.'),
                            val_if_missing=val_if_missing,
                            safe=safe, short_circuit=short_circuit)

###############################################################################
def setprop(x, prop_name, new_val):
    """
    Example:

        >>> x = { 'a' : 'before', 'b' : 'prior' }
        >>> setprop(x, 'c', 'novo')
        >>> setprop(x, 'a', 'after')
        'before'
        >>> x
        {'a': 'after', 'c': 'novo', 'b': 'prior'}

    :param x:
    :param prop_name:
    :param new_val:
    :return: the previous value of x[prop_name], if any
    """
    oldval = getprop(x, prop_name, val_if_missing=None)
    x[prop_name] = new_val
    return oldval

###############################################################################
def setprop_star(orig_x, prop_path, new_val):
    """
    Example:
    
        >>> x = { 'a' : { 'b': { 'c' : 'surprise!' } } }
        >>> setprop_star(x, "a.b", "ho-hum")
        {'c': 'surprise!'}
        >>> x
        {'a': {'b': 'ho-hum'}}
        
    :param orig_x: 
    :param prop_path: 
    :param new_val: 
    :return: the previous value of getprop_star(orig_x, prop_path), if any
    """
    def set_new_val(target_x, prop_name):
        return setprop(target_x, prop_name, new_val)

    return mutate_prop_star(orig_x, prop_path, set_new_val)

###############################################################################
def mutate_prop_star(orig_x, prop_path, mutandis):
    prop_path_components = prop_path.split('.')
    x = orig_x
    for prop in prop_path_components[:-1] :
        last_x = x
        x = getprop(x, prop)
        if x is None:
            last_x[prop] = x = {}
    return mutandis(x, prop_path_components[-1])

###############################################################################
def delprop(x, prop_name):
    """
    Example:

        >>> x = { 'a' : 'before', 'b' : 'prior' }
        >>> delprop(x, 'c')
        >>> delprop(x, 'a')
        'before'
        >>> x
        {'b': 'prior'}

    :param x:
    :param prop_name:
    :return: the previous value of x[prop_name], if any
    """
    __MISSING__ = object()
    oldval = getprop(x, prop_name, __MISSING__)
    if oldval == __MISSING__ :
        return None
    else:
        del x[prop_name]
        return oldval

###############################################################################
def delprop_star(orig_x, prop_path):
    """
    Example:

        >>> x = { 'a': { 'b': { 'c': 'surprise!' }, 'd': { 'e': 'hoo boy' } } }
        >>> delprop_star(x, "a.b")
        {'c': 'surprise!'}
        >>> x
        {'a': {'d': {'e': 'hoo boy'}}}
        >>> delprop_star(x, "a")
        {'d': {'e': 'hoo boy'}}
        >>> x
        {}
        >>> delprop_star(x, "x")
        >>> x
        {}

    :param orig_x:
    :param prop_path:
    :return: the previous value of getprop_star(orig_x, prop_path), if any
    """
    return mutate_prop_star(orig_x, prop_path, delprop)

###############################################################################
def make_property_overlayer(props_n_vals, propval_callback=None,
                            ignore_nones=False):
    """
    Return a function that takes an object, and sets ("overlays") all the given
    property:value assignments on that object.  If given a callback function,
    it will be called for each such assignment just before the assignment
    is made.

    :param props_n_vals: The dictionary of property:value assignments
    to overlay.
    :param propval_callback: f(prop_name, old_value, new_value) --> WHATEVER
    :param ignore_nones: If True, only overlay p:v where v is not None.
    :return: f(obj) --> the mutated obj
    """
    def overlay_properties(substrate_obj):
        for k, v in props_n_vals.iteritems():
            old_val = getprop_star(substrate_obj, k)
            if v is not None or not ignore_nones:
                do_if_doable(propval_callback, k, old_val, v)
                setprop_star(substrate_obj, k, v)
        return substrate_obj
    return overlay_properties


DEFAULT_HERITAGE_PATH = {'object': 'parent'}

###############################################################################
def get_inheritable_prop(obj, prop_name, heritage_path=None, no_value=None):
    """
    Returns O[prop_name], where O == the 'obj' argument or its nearest ancestor.

    "Nearest ancestor" is defined as a walk up the object graph along some set
    of object properties, which set is specified by 'heritage_path'.

    By default, each object points to its parent via the property 'parent'.
    """
    owner = get_inheritable_prop_owner(obj, prop_name, 
                                       heritage_path=heritage_path, 
                                       no_value=no_value)
    return owner and owner[prop_name]

###############################################################################
def get_composed_inheritable_prop(obj, prop_name, heritage_path=None,
                                  no_value=None, composer=None, knil=None,
                                  _climberoonie=None):
    compose = composer or compose_dicts
    climb_from = _climberoonie or \
                 make_inheritance_climber(heritage_path, no_value)
    layer_owner = get_inheritable_prop_owner(obj, prop_name,
                                             _climberoonie=climb_from)
    if layer_owner:
        layer_val = layer_owner[prop_name]
        daddy_obj = climb_from(layer_owner)
        if daddy_obj:
            return compose(get_composed_inheritable_prop(
                daddy_obj, prop_name, composer=compose, knil=knil,
                _climberoonie=climb_from),
                           layer_val)
        else:
            return layer_val
    else:
        return knil

###############################################################################
def get_inheritable_prop_owner(obj, prop_name, heritage_path=None,
                               no_value=None, _climberoonie=None):
    if obj is None:
        raise Exception("Cannot get properties from None object!")
    elif prop_name in obj:
        return obj
    else:
        climb_from = _climberoonie or \
                     make_inheritance_climber(heritage_path, no_value)
        daddy_obj = climb_from(obj)

        return daddy_obj and \
               get_inheritable_prop_owner(daddy_obj, prop_name,
                                          _climberoonie=climb_from)

###############################################################################
def make_inheritance_climber(heritage_path, no_value):
    heritage_path = heritage_path or DEFAULT_HERITAGE_PATH
    no_value = no_value or (lambda v: v is None)

    def climb_ancestry(obj):
        for iama in obj.__class__.__mro__:
            iama_type_fqn = (("" if iama.__module__ == '__builtin__' else
                              (iama.__module__ + "."))
                             +
                             iama.__name__)

            # HERITAGE_PATH keys may be <type 'type'> objects or fully
            #  qualified class names
            if iama in heritage_path:
                inherit_via = heritage_path[iama]
            elif iama_type_fqn in heritage_path:
                inherit_via = heritage_path[iama_type_fqn]
            else:
                continue

            # HERITAGE_PATH values may be functions to apply to obj,
            # or attribute names within obj
            if inspect.isroutine(inherit_via):
                parent_maybe = inherit_via(obj)
            elif isinstance(inherit_via, basestring):
                parent_maybe = getprop(obj, inherit_via)
            else:
                raise Exception("Don't understand how to inherit from"
                                " a %s via %s" % (iama_type_fqn, inherit_via))

            if parent_maybe and not no_value(parent_maybe):
                return parent_maybe

    return climb_ancestry

###############################################################################
# yoinked from :
# http://stackoverflow.com/questions/3012421/python-lazy-property-decorator

def lazyprop(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


###############################################################################
# A dash of higher-order functional programming
###############################################################################

def make_getprop(prop_name, val_if_missing=None):
    """
    Returns a procedure that returns the given property of its argument.
    """
    return lambda x: getprop(x, prop_name, val_if_missing)

###############################################################################
def prop_equals(prop_name, val):
    """
    Returns a procedure that returns True iff its argument's value for the
    given property is VAL
    """
    return lambda x: prop_name in x and x[prop_name] == val

###############################################################################
def is_equal(y):
    return lambda x: x == y

###############################################################################
def is_in(y_ls):
    return lambda x: x in y_ls

###############################################################################
def is_not_in(y_ls):
    return lambda x: x not in y_ls

###############################################################################
def find(f, seq, exc_msg=None):
    """
    Return first item in sequence for which f(item) tests True.
    """
    result_ish = or_map(lambda x: f(x) and [x], seq, exc_msg=exc_msg)
    return result_ish[0] if result_ish else None

###############################################################################
def or_map(f, seq, exc_msg=None):
    """
    Return first result f(item) in sequence that tests True;
    else returns the last result f(item)
    """
    maybe_result = None
    for item in seq:
        maybe_result = f(item)
        if maybe_result:
            return maybe_result

    if exc_msg:
        raise Exception(exc_msg)
    else:
        return maybe_result

###############################################################################
def filter_map(f, *arg_lists):
    return filter(None, map(f, *arg_lists))

###############################################################################
def partition(has_it, items):
    """
    Returns a 2-tuple containing the list of items satisfying
    and the list of items not satisfying the predicate has_it.
    """
    haves = []
    have_nots = []
    for i in items:
        if has_it(i):
            haves.append(i)
        else:
            have_nots.append(i)
    return ( haves, have_nots )

###############################################################################
def identity(thing):
    """
    Try writing
        func = optional_func or lambda x: x
    Oh!  You can't!  Ok, so use this as the second clause of the disjunction.
    :param thing: the thing
    :return: that same thing
    """
    return thing


#################### dict vs. tuple #####################################
##
## Transform simple dict objects into something hashable, and back again.
##
## Not indicated for use with dicts having complex (unhashable) values.
##

def dict_to_tuple(sd):
    """
    >>> dict_to_tuple({'a':1, 'b':2})
    (('a', 1), ('b', 2))
    >>> dict_to_tuple({'a':1, 'b':2}) == dict_to_tuple({'b':2, 'a':1})
    True
    """
    return tuple((k, v) for k, v in sorted(sd.items()))

###############################################################################
def tuple_to_dict(tp):
    """
    >>> x = tuple_to_dict((('a', 1), ('b', 2)))
    >>> x['b']
    2
    """
    return dict(tp)


#################### compose_dicts ######################################

def compose_dicts(*dict_list):
    """
    >>> x = compose_dicts({'a':1, 'b':2}, {'c':3, 'b':52})
    >>> dict_to_tuple(x)
    (('a', 1), ('b', 52), ('c', 3))
    """
    result = {}
    for this_dict in filter(None, dict_list):
        result.update(this_dict)
    return result


#################### filter_dict ########################################

def filter_dict(the_dict, kv_filter=None, key_filter=None, value_filter=None):
    """
    >>> ddd = {'a':1, 'b':2, 'c':3, 4:4, 5:5}
    >>> filter_dict(ddd, lambda key, val: key == val)
    {4: 4, 5: 5}
    >>> filter_dict(ddd, key_filter=is_equal('b'))
    {'b': 2}
    >>> filter_dict(ddd, value_filter=lambda v: v % 2)
    {'a': 1, 'c': 3, 5: 5}
    >>> filter_dict(ddd, kv_filter=operator.eq)
    {4: 4, 5: 5}
    >>> filter_dict(ddd, value_filter=lambda v: int(v) % 2,
    kv_filter=operator.eq)
    {5: 5}
    """
    fk, fv, fkv = map(is_applicable, (key_filter, value_filter, kv_filter))
    is_cool = lambda k, v: ((not fk or key_filter(k)) and
                            (not fv or value_filter(v)) and
                            (not fkv or kv_filter(k, v)))
    return { k : v for k, v in the_dict.iteritems() if is_cool(k, v) }


#################### extract_map ########################################

def extract_map( things,
                 key_extractor=None,
                 keys_extractor=None,
                 value_extractor=None,
                 values_extractor=None,
                 value_accumulator=None,
                 thing_summarizer=str,
                 knil=None,
                 exclude_none_key=False,
                 result=None ):
    result = {} if result is None else result
    thing2vals = _make_thing2stuffs( value_extractor, values_extractor )
    thing2keys = _make_thing2stuffs( key_extractor, keys_extractor )
    accumulate = ( value_accumulator if value_accumulator is not None else
                   lambda v_old, v_new: v_old + v_new )
    for thing in things:
        try:
            vals = thing2vals( thing )
            for k in thing2keys( thing ) :
                if k is None and exclude_none_key:
                    continue
                for v in vals :
                    if k in result :
                        result[k] = accumulate( result[k], v )
                    elif knil is not None:
                        result[k] = accumulate( knil, v )
                    else :
                        result[k] = v
        except Exception, e:
            (et, ev, etb) = sys.exc_info()
            print "Problem with %s : %s" % (thing_summarizer(thing), str(e))
            raise e, None, etb
    return result


def _make_thing2stuffs( stuff_ex, stuffs_ex ):
    if stuffs_ex is not None:
        return stuffs_ex
    elif stuff_ex is not None:
        return lambda thng: listify( stuff_ex( thng ) )
    else:
        return listify


###############################################################################
def listify( thing ):
    return [ thing ]


###############################################################################
# Counting: what numbers were made for
###############################################################################

def next_sequence_value(vals, default_stride=1):
    svals = sorted(vals)
    vfrom = svals[:-1]
    vto = svals[1:]
    strides = map(operator.sub, vto, vfrom)
    unique_strides = set(strides)
    if len(unique_strides) == 1 :
        return svals[-1] + (unique_strides.pop() or default_stride)
    else:
        return svals[-1] + default_stride


###############################################################################
# Robustification: functions & decorators to auto-retry failure-prone
# operations
###############################################################################

from functools import wraps, update_wrapper


def robustify(**retry_kwargs):
    """
    This decorator factory produces a decorator which wraps its decorated
    function with retry_till_done() (q.v.), invoked according to the given
    optional keyword arguments.

    >>> y = 3
    >>> #noinspection PyUnresolvedReferences
    >>> @robustify(max_attempts=3, failure_val="drat")
    ... def foo(x):
    ...     print "i think i can"
    ...     global y
    ...     y += x
    ...     if y < 10:
    ...         raise Exception("not there yet")
    ...     return y
    >>> foo(1)
    i think i can
    i think i can
    i think i can
    'drat'
    >>> foo(3)
    i think i can
    i think i can
    12

    Robustification ==> the Exception was never permitted to escape.
    """
    def robustificator(f):
        @wraps(f)
        def f_robustified(*args, **kwargs):
            return retry_till_done(lambda: f(*args, **kwargs),
                                   **retry_kwargs)
        return f_robustified
    return robustificator


###############################################################################
def robustify_methods(exclude=None, include=None, **retry_kwargs):
    """
    Class decorator which "robustifies" all methods in the decorated class,
    according to the retry_kwargs given.

    :param list exclude: names of methods to leave un-robustified
    :param list include: names of methods to explicitly robustify
    (excluding others)
    :param dict retry_kwargs:  anything you could pass to retry_till_done()
    (q.v.)
    :return: the robustified class.
    """
    original_methods = {}

    def method_robustificator(methname):
        orig_f = original_methods[methname]

        def f_robustified(*args, **kwargs):
            do_the_stuff = lambda: orig_f(*args, **kwargs)
            return retry_till_done(do_the_stuff, **retry_kwargs)
        return update_wrapper(f_robustified, orig_f)

    def class_robustificator(c):
        for method_name in filter(lambda n: hasmethod(c, n), dir(c)):
            if exclude and method_name in exclude:
                continue
            if include and method_name not in include:
                continue
            oldmeth = getattr(c, method_name)
            original_methods[method_name] = oldmeth   # pyclosure ftl
            newmeth = method_robustificator(method_name)
            newmeth.__name__ = method_name + "_robustified"
            newmeth.__doc__ = (None if oldmeth.__doc__ is None else
                               oldmeth.__doc__ + "\n (Robustified)")
            setattr(c, method_name, newmeth)
        return c
    return class_robustificator


###############################################################################
class CaUGHTaBULLET(object):
    def __init__(self, exc):
        self.missile = exc

###############################################################################
def do_if_doable_bulletproof(doable_or_none, description, *args, **kwargs):
    desc = description or "doable"
    try:
        return do_if_doable(doable_or_none, *args, **kwargs)
    except Exception, e:
        logger.error("Caught a bullet while doing %s : %s" %
                     (desc, e))
        return CaUGHTaBULLET(e)


###############################################################################
def retry_till_done(do_it, is_done=None, is_good=None,
                    max_wait_in_secs=None, max_attempts=None,
                    do_on_exception=True, do_on_error=None,
                    success_val=None, failure_val=False,
                    do_on_failure=None,
                    do_between_attempts=None, retry_interval=5,
                    backoff=1,
                    meta_args=None):
    """
    Repeatedly calls do_it() until it succeeds or bails.

    Succeeds ==> does not raise an Error or Exception, and
                 is_done() returns True (if is_done is given), and
                 is_good(_last_result_) returns True (if is_good is given).
              ==> Returns success_val if given (non-None), or else
                  [default] returns the value from the last do_it() call,
                  aka _last_result_ .

    Bails ==> an Error or Exception is not handled, or (if they are given)
              one of max_attempts or max_wait_in_secs has been exceeded.
              ==> Runs do_on_failure() if it is given (and runnable), and then
                  (if that didn't raise an Exception, which it is allowed
                  to do),
              ==> If the ultimate do_it() raised an Exception, and
              do_on_failure is an Exception,
                  re-raise the Exception from do_it; otherwise,
              ==> Returns the last do_it() return if failure_val is None,
              or else
                  [default] returns failure_val itself -- by default: False.

    Limits: max_attempts == will call do_it() AT MOST this many times
            max_max_wait_in_secs == will stop calling do_it() once this many
                                    seconds have elapsed since the initial attempt
            These limits are enforced only when supplied values GREATER THAN 0.
            DEFAULTS: max_attempts == max_max_wait_in_secs == 0
                      ==> will keep trying FOREVER by default.

    Waits retry_interval seconds between do_it() attempts, after having
    run do_between_attempts() if that function is given.
    (Wait interval is multiplied by backoff [default: 1] on each iteration.)

    Errors and Exceptions are caught and handled according to
    do_on_error and do_on_exception, respectively.  If the value is:
              True            ==> keep on truckin' (silently mask)
              None or False   ==> re-raise
              a function FOO  ==> run FOO() and continue

    meta_args is a preview feature, whose use is beyond the scope of this
    header comment -- employ it at your own risk.

    The default behavior is to re-raise Errors, and silently mask Exceptions.
    """
    num_attempts = 0
    start_time = now()
    succeeded = False
    last_result = None
    last_exception = None
    next_wait = 5
    th_locals = locals()

    meta_args = meta_args or {}
    args_dict = inspect.getargvalues(inspect.currentframe()).locals
    handler_arg_names = ['do_it', 'is_done', 'is_good', 'do_on_exception',
                         'do_on_error', 'do_on_failure', 'do_between_attempts']
    (do_it, is_done, is_good, do_on_exception,
     do_on_error, do_on_failure, do_between_attempts) = \
        (repackage(args_dict[fn_name], meta_args[fn_name])
         if fn_name in meta_args
         else args_dict[fn_name]
         for fn_name in handler_arg_names)

    next_wait = retry_interval

    def had_enough():   # n.b. never satisfied if max_FOOs are <= 0 or None.
        return ((0 < max_attempts <= num_attempts) or
                (0 < max_wait_in_secs <= now() - start_time))

    while not had_enough():
        # noinspection PyBroadException
        try:
            if num_attempts > 0:
                betweentimes_result = \
                    do_if_doable_bulletproof(do_between_attempts,
                                             "do_between_attempts")
                sleep(next_wait)
                next_wait *= backoff
            num_attempts += 1
            last_exception = None
            last_result = do_it()
            succeeded = ((is_done is None or is_done()) and
                         (is_good is None or is_good(last_result)))
            if succeeded:
                break
        except Exception, e:
            last_exception = e
            handle_via(do_on_exception, e)
        except:
            handle_via(do_on_error)

    if succeeded :
        return last_result if success_val is None else success_val
    else :
        do_if_doable(do_on_failure)
        if last_exception and isinstance(do_on_failure, Exception):
            reraise(last_exception)
        return last_result if failure_val is None else failure_val


def do_if_doable(doable_or_none, *args, **kwargs):
    """
    If the first argument is a procedure, return the result of applying it
    to the supplied arguments. Otherwise - and this is important - return None.
    """
    if is_applicable(doable_or_none):
        return doable_or_none(*args, **kwargs)


def is_applicable(doable_or_none):
    return type(doable_or_none) in [ types.FunctionType, types.MethodType,
                                     types.BuiltinFunctionType,
                                     types.BuiltinMethodType ]


def handle_via(do_handling, *args):
    if isinstance(do_handling, bool) and do_handling:
        pass            # True ==> persevere through all that
    elif not do_handling:
        raise           # None/False ==> let that propagate back to caller
    else:
        do_if_doable(do_handling, *args)


def repackage(func, meta_formals, frame_offset=1):
    """
    This function is not suitable for any given purpose, and should be used
    by no one.  You have been warned.
    """
    def arg_val(formal, locals_dict):
        if formal in locals_dict:
            return locals_dict[formal]
        else:
            raise Exception("Bad meta arg '%s'" % formal)

    @wraps(func)
    def wrapped_func(*args):
        th_frames = inspect.getouterframes(inspect.currentframe())
        th_locals = inspect.getargvalues(th_frames[frame_offset][0]).locals
        try:
            meta_actuals = [arg_val(formal, th_locals) for
                            formal in meta_formals]
            return func(*(args + tuple(meta_actuals)))
        finally:
            # per http://docs.python.org/2/library/inspect.html#the-interpreter-stack
            del th_frames

    return wrapped_func


def do_or_die(thunk, death_rattle=None, **retry_params):
    miserable_failure = "-- lose --"
    retry_params['failure_val'] = miserable_failure
    outcome = retry_till_done(thunk, **retry_params)
    if outcome is miserable_failure:
        die_with_err(death_rattle or miserable_failure)
    else:
        return outcome


###############################################################################
def wait_for(predicate, timeout=None, on_wait=None, **kwargs):
    """
    Waits until predicate() is true, periodically evaluating that function.

    on_wait(), if given, will be called after each unsuccessful evaluation.

    timeout ==> maximum number of seconds to wait before succeeding or else
                raising an Exception

    Accepts all other keyword arguments accepted by retry_till_done(), to which
    it delegates.
    """
    rtd_kwargs = { k : v for k, v in kwargs.iteritems() }

    def specify_if_needed(kwarg_key, kwarg_val):
        if kwarg_key not in rtd_kwargs :
            rtd_kwargs[kwarg_key] = kwarg_val

    specify_if_needed('success_val', True)
    specify_if_needed('failure_val', False)
    specify_if_needed('max_wait_in_secs', timeout)
    specify_if_needed('do_between_attempts', on_wait)
    return retry_till_done(lambda: "yay", is_done=predicate, **rtd_kwargs)


def raise_or_return(result):
    if isinstance(result, Exception):
        raise result
    else:
        return result


def idle_messenger(what="the desired condition to obtain", logger=None):
    """ :return a nice thunk for passing as on_wait / do_between_attempts """
    the_logger = logger_for_sure(logger)
    pyclosure = { "t0" : None }

    def your_call_is_important_to_us():
        if pyclosure['t0']:
            how_long = " (%.0f secs now)" % (now() - pyclosure['t0'])
        else:
            pyclosure['t0'] = now()
            how_long = ""
        the_logger.info("Waiting for %s%s..." % (what, how_long))

    return your_call_is_important_to_us


def raising_only(*exception_types):
    """ :return a nice proc for passing as do_on_exception, which re-raises an
                exception only if it is an instance of one of the given types.
    """
    def raise_if_matching(exception):
        if find(lambda t: isinstance(exception, t), exception_types):
            reraise(exception)  # tries to retain original traceback stack
    return raise_if_matching


def ex_collector(receptacle, what="the desired condition to obtain",
                 logger=None,
                 collection_method='append'):
    """ :return a nice proc for passing as do_on_exception """
    the_logger = logger_for_sure(logger)
    if not receptacle:
        collect = lambda exc: 99
    elif is_applicable(collection_method):
        collect = collection_method
    elif hasattr(receptacle, collection_method):
        collect = getattr(receptacle, collection_method)
    else:
        raise Exception("Cannot collect exceptions into this: %s" % receptacle)

    def got_a_live_one_here_charlie(exc):
        if what:
            the_logger.info("Exception while waiting for %s :"
                            " %s" % (what, exc))
        collect(exc)

    return got_a_live_one_here_charlie


def summarize_exceptions(exes):
    if exes:
        return "%d exceptions including %s" % (len(exes), exes[0])
    else:
        return "no exceptions"


def describe_wait_params(waiting_kwargs):
    def constraint_desc(prop_name, noun):
        val = getprop(waiting_kwargs, prop_name)
        return "%d %s" % (val, noun) if val else ""
    return " or ".join(filter_map(constraint_desc,
                                  ['max_attempts', 'timeout',
                                   'max_wait_in_secs'],
                                  ["attempts", "seconds", "seconds"]))


def await_condition(is_as_desired, what_do_we_want="the thing we want",
                    wait_for_kwargs=None):
    waiting_defaults = {'max_attempts': 100,
                        'retry_interval': 6,
                        'on_wait': idle_messenger(what_do_we_want, logger)}
    waiting_kwargs = compose_dicts(waiting_defaults, wait_for_kwargs or {})
    result = wait_for(is_as_desired, **waiting_kwargs)
    if not result:
        raise Exception("After %s, unable to wait any longer for %s!" %
                        (describe_wait_params(waiting_kwargs),
                         what_do_we_want))
    return result


###############################################################################
# forbidding recursion - there may be a reason why
###############################################################################

def norecurse(f):
    pyclosure = { 'calls' : 0 }

    @wraps(f)
    def f_one_shot(*args, **kwargs):
        if pyclosure['calls'] :
            # I said no recurse!
            return
        else:
            try:
                pyclosure['calls'] += 1
                return f(*args, **kwargs)
            finally:
                pyclosure['calls'] -= 1
    return f_one_shot



###############################################################################
# pyflection
###############################################################################

def is_iterable(a_list_maybe):
    return isinstance(a_list_maybe, collections.Iterable)

###############################################################################
# credit:
#   http://stackoverflow.com/questions/1091259/how-to-test-if-a-class-attribute-is-an-instance-method
#
def hasmethod(kls, name):
    return hasattr(kls, name) and isinstance(getattr(kls, name),
                                             types.MethodType)

###############################################################################
def hasroutine(obj, name):
    return hasattr(obj, name) and inspect.isroutine(getattr(obj, name))


