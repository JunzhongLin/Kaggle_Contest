
def test(x, e=1, **kwargs):
    print(kwargs)

kw = {
    'a': 1,
    'b': 2,
    'c': 3
}

test(3,**kw)