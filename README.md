# Generate Globs

Generate glob expressions by example. 


## Usage

Generate a list of globs that match all elements of `whitelist` and none of `blacklist`.

```
>>> import fnmatch
>>> # whitelist, blacklist = [], []
>>> globs = generate_globs(whitelist, blacklist)
```

`generate_globs` generates globs that fulfill both assertions:

```
>>> assert all([any([fnmatch.filter([white], glob) for glob in globs]) for white in
>>>            whitelist])
>>> assert not any([fnmatch.filter(blacklist, glob) for glob in globs])
```

For example:

```
>>> generate_globs(whitelist=['data1', 'data2', 'data3'], blacklist=['admin']) 
{['data*']}

>>> generate_globs(whitelist=['a', 'b', 'c'], blacklist=['d']) 
{['[a-c]']}
```


Returns an empty list, if `whitelist` is empty.

## Limitation

Generating good globs for arbitrary input is hard, thus only expect decent globs for "friendly" input.
Also, don't use it for user input.


## Running the tests    
    py.test -v  test_generate_globs.py