def new_frame(text):
    import sys
    sys.stdout.write('\r' + text)
    sys.stdout.flush()


def clear_frame():
    import os
    os.system('clear')

