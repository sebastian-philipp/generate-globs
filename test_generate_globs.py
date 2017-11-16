"""
 *   Copyright (c) 2017 SUSE LLC
 *
 * This file is covered by the LICENSING file in the root of this project.
"""

import pytest

from generate_globs import Glob, generate_globs, GlobSolution


def g(single_string):
    return Glob.from_string(single_string)


def gs(strings):
    """Generate a GlobSolution for `strings`"""
    if isinstance(strings, list):
        return GlobSolution({g(s) for s in strings})
    else:
        return GlobSolution(g(strings))


def fzs(*args):
    """
    Generate a set of frozensets of args. This is needed, because set cannot be a member of a set.

    >>> fzs({'a', 'b'}, 'c') == {frozenset({'a', 'b'}), frozenset({'c'})}
    """
    return {frozenset(s) if isinstance(s, set) else frozenset({s}) for s in args}


def gs_set_to_str_set(gss):
    """
    Converts a set of glob solutions to a set of frozensets of strings. This is needed, because
    there is no way to generate a complex Glob from a complex glob

    >>> gs_set_to_str_set({gs('a'), gs('b', 'c')}) == {frozenset({'a'}, frozenset{'b', 'c'})}
    """
    return set(map(GlobSolution.str_set, gss))


def test_glob_base():
    assert len({g(''), g('')}) == 1

    star = Glob([(Glob.T_Any,)])
    assert star + star == star


def test_string():
    assert g('abc'), Glob([(1, 'a'), (1, 'b') == (1, 'c')])
    assert g('') == Glob([])
    assert str(g('aa')) == 'aa'


def test_glob_merges():
    assert g('aa').commonsuffix(g('aba')) == Glob([(1, 'a')])

    assert str(g('aa').merge_any(g('ab'))) == '*'
    assert list(map(str, g('a').merge_one(g('b')))) == ['?']
    assert list(map(str, g('a').merge_range(g('b')))) == ['[ab]']

    assert set(map(str, g('aa').merge_all(g('ab')))) == {'a[ab]', 'a*', 'a?'}
    assert list(map(str, g('').merge_all(g('a')))) == ['*']
    assert set(map(str, g('a').merge_all(g('bc')))) ==  {'[ab]*', '*', '?*'}

    assert gs_set_to_str_set(g('a').merge(g('bc'), [])) == fzs('*', '?*', '[ab]*')
    assert gs_set_to_str_set(g('a').merge(g('bc'), ['ac'])) == fzs({'a', 'bc'})


def test_globs_merge():

    assert gs_set_to_str_set(gs('a').merge_solutions(gs('b'), [])) == fzs('*', '?', '[ab]')
    assert gs_set_to_str_set(gs('a').merge_solutions(gs('b'), ['c'])) == fzs('[ab]')

    assert gs_set_to_str_set(gs(['ab', 'bc']).merge_solutions(gs('ac'), [])) == fzs({'ab', '*c'},
                                                                                    {'ab', '?c'},
                                                                                    {'a*', 'bc'},
                                                                                    {'a?', 'bc'})

    with pytest.raises(ValueError):
        gs(['ab', 'bc']).merge_solutions(gs('ac'), ['ac'])
    assert gs_set_to_str_set(gs(['a', 'bb']).merge_solutions(gs('ccc'), ['ab', 'ac', 'bc'])) == fzs({'a', 'bb', 'ccc'})

    one_any = Glob([(Glob.T_One,), (Glob.T_Any, )])
    one_one_any = Glob([(Glob.T_One,), (Glob.T_One,), (Glob.T_Any, )])
    assert gs_set_to_str_set(one_any.merge(one_one_any, [])) == fzs('?*')


def test_gen_globs():
    assert generate_globs(['a', 'b', 'c'], []) == frozenset(['*'])
    assert generate_globs(['a', 'b', 'c'], ['d']) == frozenset(['[a-c]'])
    assert generate_globs(['a', 'b', 'd'], ['c']) == frozenset(['[abd]'])
    assert generate_globs(['data1', 'data2', 'data3'], ['admin']) == frozenset(['data*'])
    assert generate_globs(['data1', 'data2', 'data3'], ['admin', 'data4']) == frozenset(['data[1-3]'])
    assert generate_globs(['data1', 'data2', 'data3'], ['admin', 'data1x']) == frozenset(['data?'])
    assert generate_globs(['ab', 'bc', 'ac'], ['bb']) == frozenset(['ab', '*c'])

    with pytest.raises(ValueError):
        generate_globs(['a', 'b'], ['a'])

    assert generate_globs(['x1x', 'x2x', 'x3x'], ['xxx']) == frozenset(['x[1-3]x'])
    assert generate_globs(['x1y3z', 'x2y2z', 'x3y1z'], ['xxyzz']) == frozenset(['x[1-3]y[1-3]z'])
    wl = ['data1', 'data2', 'mon1', 'mon2', 'mon3', 'igw1', 'igw2']
    bl = ['client1', 'client2', 'admin1', 'admin2', 'rgw1', 'rgw2']

    assert generate_globs(wl, bl) == frozenset(['[dim][ago][ntw]*', 'data1'])


# pytest-quickcheck is broken in version 0.8.3: Don't expect any useful tests here.
@pytest.mark.randomize(
    whitelist={str},
    blacklist={str},
    ncalls=3
)
def test_generate_dict(whitelist, blacklist):
    generate_globs(whitelist, blacklist - whitelist)
