def new_frame(text):
    import sys
    sys.stdout.write('\r' + text)
    sys.stdout.flush()


def clear_frame():
    import os
    os.system('clear')


def progress(num):
    import sys
    step = 1
    perc = '%3d%%' % int(num * 100)
    bar = int(num * 100 / step)
    space = int(100 / step) - bar
    sys.stdout.write('\r[' + ('=' * bar) + (' ' * space) + '] ' + perc)