"""
 *   Copyright (c) 2017 SUSE LLC
 *
 * This file is covered by the LICENSING file in the root of this project.
"""

import fnmatch
from functools import total_ordering
from itertools import product

from os.path import commonprefix


def generate_globs(whitelist, blacklist):
    """
    Generate a list of globs that match all elements of `whitelist` and none of `blacklist`.

    >>> import fnmatch
    >>> whitelist, blacklist = [], []
    >>> globs = generate_globs(whitelist, blacklist)
    >>> assert all([any([fnmatch.filter([white], glob) for glob in globs]) for white in
    >>>            whitelist])
    >>> assert not any([fnmatch.filter(blacklist, glob) for glob in globs])

    For example:

    >>> generate_globs(whitelist=['data1', 'data2', 'data3'], blacklist=['admin'])
    {['data*']}

    >>> generate_globs(whitelist=['a', 'b', 'c'], blacklist=['d'])
    {['[a-c]']}


    Returns an empty list, if `whitelist` is empty.

    :type whitelist: iterable[str]
    :type blacklist: iterable[str]
    :rtype: frozenset[str]
    :raise ValueError: If white and blacklist overlap.
    """

    def merge_globs_rec(globs):
        """
        merge_globs_rec merges two glob proposals in a tree-like way.

        :type globs: list[str]
        :rtype: set[GlobSolution]
        """
        if len(globs) == 1:
            return {GlobSolution(Glob.from_string(globs[0]))}

        first_half = globs[:len(globs) // 2]
        second_half = globs[len(globs) // 2:]

        return merge_two_globs_proposals(merge_globs_rec(first_half),
                                         merge_globs_rec(second_half),
                                         blacklist)

    if not whitelist:
        return []

    res = merge_globs_rec(list(whitelist))
    best_globs = sorted(res, key=lambda s: s.complexity())[0]

    # TODO: remove, if your confidence level in generate_globs() is high enough.
    assert all([any([fnmatch.filter([white], glob) for glob in best_globs.str_set()]) for white in
                whitelist])
    assert not any([fnmatch.filter(blacklist, glob) for glob in best_globs.str_set()])

    return best_globs.str_set()


def merge_two_globs_proposals(ls, rs, blacklist):
    """
    Generates a set of all merged glob proposals. All results match the union of ls and rs.

    :type ls: set[GlobSolution]
    :type rs: set[GlobSolution]
    :type blacklist: list[str]
    :rtype: set[GlobSolution]
    """

    proposals = set()
    for l, r in product(ls, rs):
        proposals.update(l.merge_solutions(r, blacklist))
    return set(sorted(proposals, key=lambda s: s.complexity())[:3])


class GlobSolution(object):
    """Represents one solution of multiple globs"""

    def __init__(self, globs):
        """:type globs: iterable[Glob] | Glob"""
        if isinstance(globs, Glob):
            self.globs = frozenset({globs})
        elif isinstance(globs, frozenset):
            self.globs = globs
        elif isinstance(globs, set):
            self.globs = frozenset(globs)
        else:
            assert False

    def merge_solutions(self, other, blacklist):
        """
        Generate lots of solutions for these two solutions. All results match both input solutions.

        :type other: GlobSolution
        :type blacklist: list[str]
        :rtype: set[GlobSolution]
        """

        ret = []
        for l, r in product(self.globs, other.globs):
            merges = l.merge(r, blacklist)
            for merge in merges:
                merge_set = set(merge.globs)

                self_no_l = set(self.globs).difference({l})
                other_no_r = set(other.globs).difference({r})

                merge_set.update(self_no_l)
                merge_set.update(other_no_r)
                ret.append(GlobSolution(merge_set))

        return set(sorted(ret, key=lambda s: s.complexity())[:4])

    def __str__(self):
        return 'GlobSolution({})'.format(map(str, self.globs))

    def complexity(self):
        return sum((8 + g.complexity() for g in self.globs))

    def __hash__(self):
        return hash(self.globs)

    def __eq__(self, other):
        return self.globs == other.globs

    def str_set(self):
        return frozenset(map(str, self.globs))

    def __repr__(self):
        return str(self)


@total_ordering
class Glob(object):
    T_Char = 1  # Matches a specific char "x"
    T_Any = 2  # Matches any string "*"
    T_One = 3  # Matches one character "?"
    T_Range = 4  # Matches a set of chars "[a-z1-5]"

    def __init__(self, elems=None):
        """
        `elems` is a list of glob-elements. Each glob-element is one of:
            1. A pair of T_Char and a char ,e.g. `(T_Char, 'x')`
            2. A one-elemnt tuple of T_Any, e.g. `(T_Any, )`
            3. A one-elemnt tuple of T_One, e.g. `(T_One, )`
            3. A pair of T_Range and a set of chars, e.g. `(T_Range, set('ab01'))`

        Note, consecutive elems of `T_Any` are invalid.

        :type elems: list[tuple] | tuple[tuple]
        """
        if elems is None:
            self.elems = tuple()
        elif isinstance(elems, Glob):
            self.elems = elems.elems
        elif isinstance(elems, list):
            self.elems = tuple(elems)
        else:
            assert isinstance(elems, tuple)
            self.elems = elems

    @staticmethod
    def from_string(s):
        return Glob([(Glob.T_Char, c) for c in s])

    @staticmethod
    def make_range_string(range_set):
        """
        Generates strings like "a-c" or "abde" or "1-5e-g"

        :type range_set: set[str]
        """

        sorted_list = sorted(map(ord, range_set))
        chunks = _split_chunks(sorted_list)
        return ''.join([
            ''.join(map(chr, chunk)) if len(chunk) <= 2 else '{}-{}'.format(
                chr(chunk[0]), chr(chunk[-1]))
            for chunk
            in chunks
            ])

    def __str__(self):
        def mk1(elem):
            """:type elem: tuple"""
            return {
                Glob.T_Char: lambda: elem[1],
                Glob.T_Any: lambda: '*',
                Glob.T_One: lambda: '?',
                Glob.T_Range: lambda: '[{}]'.format(self.make_range_string(elem[1])),
            }[elem[0]]()
        return ''.join(map(mk1, self.elems))

    def __getitem__(self, val):
        ret = self.elems.__getitem__(val)
        if isinstance(ret, list):
            return Glob(ret)
        if isinstance(ret, tuple) and (not ret or isinstance(ret[0], tuple)):
            return Glob(ret)
        if isinstance(ret, Glob):
            return ret
        assert isinstance(ret, tuple)
        return ret

    def __eq__(self, other):
        return self.elems == other.elems

    def __lt__(self, other):
        return self.elems < other.elems

    def __hash__(self):
        return hash(self.elems)

    def complexity(self):
        """Returns a complexity indicator. Simple glob expressions are preferred."""
        def complexity1(index, e):
            ret = {
                Glob.T_Char: lambda: 0.0,
                Glob.T_Any: lambda: 1.0,
                Glob.T_One: lambda: 2.0,
                Glob.T_Range: lambda: max(3, len(self.make_range_string(e[1]))),  # prefer small
            }[e[0]]()
            if e[0] != Glob.T_Char and index != len(self) - 1:
                ret += 0.5  # Prefer globing last character
            return ret

        return sum((complexity1(index, elem) for index, elem in enumerate(self.elems)))

    def merge(self, r, blacklist):
        """
        Merges this glob with `r` by creating multiple solutions. Filters all solutions that
        violate the blacklist.

        :type r: Glob
        :type blacklist: list[str]
        :rtype: set[GlobSolution]
        :raise ValueError: If either self or r matches the blacklist.
        """
        for e in [self, r]:
            if any((fnmatch.fnmatch(black, str(e)) for black in blacklist)):
                raise ValueError('Glob "{}" already matches blacklist.'.format(e))

        merged = self.merge_all(r)
        ok = {e for e in merged if not any((fnmatch.fnmatch(black, str(e)) for black in blacklist))}
        ok = sorted(ok, key=Glob.complexity)[:3]
        ret = {GlobSolution(e) for e in ok} if ok else {GlobSolution({self, r})}
        return ret

    def merge_all(self, r):
        """
        Generates a set of all possible merges between self and r. Can be empty.

        :type r: Glob
        :rtype: set[Glob]"""
        if self == r:
            return {self}
        if not self or not r:
            return {self.merge_any(r)}

        prefix = self.commonprefix(r)
        suffix = self[len(prefix):].commonsuffix(r[len(prefix):])
        mid_l = self[len(prefix):len(self)-len(suffix)]
        mid_r = r[len(prefix):len(r)-len(suffix)]

        def fix(merged):
            if merged is None:
                return None
            return prefix + merged + suffix

        ret = set()
        ret.add(fix(mid_l.merge_any(mid_r)))

        one_merged = mid_l.merge_one(mid_r)
        if one_merged is not None:
            ret.update(map(fix, one_merged))

        range_merged = mid_l.merge_range(mid_r)
        if range_merged is not None:
            ret.update(map(fix, range_merged))

        if None in ret:
            ret.remove(None)
        return ret

    def __add__(self, other):
        if self.elems[-1:] == ((Glob.T_Any, ), ) and other.elems[:1] == ((Glob.T_Any, ), ):
            return Glob(self.elems + other.elems[1:])
        return Glob(self.elems + other.elems)

    def __nonzero__(self):
        return bool(self.elems)

    def __len__(self):
        return len(self.elems)

    def merge_any(self, other):
        if not self and not other:
            return Glob()
        return Glob([(Glob.T_Any, )])

    def merge_one(self, other):
        """
        :type other: Glob
        :rtype: set[Glob] | None
        """

        length = min(len(self), len(other))
        ranges = [_merge_one(e1, e2) for e1, e2 in zip(self[:length], other[:length])]
        if any([range_elem is None for range_elem in ranges]):
            return None
        ends = self[length:].merge_all(other[length:])
        return {Glob(ranges) + Glob(merged.elems) for merged in ends}

    def merge_range(self, other):
        """
        :type other: Glob
        :rtype: set[Glob] | None
        """
        def combine_range_char(range_elem_1, char_elem):
            return Glob.T_Range, frozenset(range_elem_1[1].union({char_elem[1]}))

        def combine_ranges(range_elem_1, range_elem_2):
            return Glob.T_Range, frozenset(range_elem_1[1].union(range_elem_2[1]))

        def one(elem1, elem2):
            """
            :type elem1: tuple
            :type elem2: tuple
            :rtype: tuple | None
            """
            t_1 = elem1[0]
            t_2 = elem2[0]
            if t_1 == Glob.T_Char and t_2 == Glob.T_Char:
                if elem1[1] != elem2[1]:
                    return Glob.T_Range, frozenset({elem1[1], elem2[1]})
                else:
                    return Glob.T_Char, elem2[1]
            if t_1 == Glob.T_Range and t_2 == Glob.T_Char:
                return combine_range_char(elem1, elem2)
            if t_1 == Glob.T_Char and t_2 == Glob.T_Range:
                return combine_range_char(elem2, elem1)
            if t_1 == Glob.T_Range and t_2 == Glob.T_Range:
                return combine_ranges(elem1, elem2)
            if (t_1 == Glob.T_Range and t_2 == Glob.T_One) or (
                    t_1 == Glob.T_One and t_2 == Glob.T_Range):
                return Glob.T_One,
            return None

        length = min(len(self), len(other))
        ranges = [one(e1, e2) for e1, e2 in zip(self[:length], other[:length])]
        if any([range_elem is None for range_elem in ranges]):
            return None
        ends = self[length:].merge_all(other[length:])
        return {Glob(ranges) + Glob(merged.elems) for merged in ends}

    def commonsuffix(self, other):
        return self[::-1].commonprefix(other[::-1])[::-1]

    def commonprefix(self, other):
        return Glob(commonprefix([self, other]))

    def __repr__(self):
        return 'Glob(({}))'.format(', '.join([repr(elem) for elem in self.elems]))


def _split_chunks(l):
    """
    Generates a list of lists of neighbouring ints. `l` must not be empty.

    >>> _split_chunks([1,2,3,5,6,7,9])
    [[1,2,3],[5,6,7],[9]]

    :type l: list[int]
    :rtype list[list[int]]
    """
    ret = [[l[0]]]
    for c in l[1:]:
        if ret[-1][-1] == c - 1:
            ret[-1].append(c)
        else:
            ret.append([c])
    return ret


def _merge_one(elem1, elem2):
    """
    :type elem1: tuple
    :type elem2: tuple
    :rtype: tuple | None
    """
    t_1 = elem1[0]
    t_2 = elem2[0]
    if t_1 == Glob.T_Char and t_2 == Glob.T_Char:
        if elem1[1] != elem2[1]:
            return Glob.T_One,
        else:
            return Glob.T_Char, elem2[1]
    if Glob.T_Any in [t_1, t_2]:
        return None
    return Glob.T_One,