def test(a,b,c):
    return a+b+c


if __name__ =='__main__':
    kwargs= {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4
    }

    print(test(**kwargs))