from pynput.mouse import Listener

def on_move(x,y):
    print(f"\rCurrent Position X={x}, Y={y}", end="")

listener = Listener(on_move=on_move)
listener.start()
listener.join()