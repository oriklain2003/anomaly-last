
import sys, time, math, os, shutil, colorsys, ctypes

MESSAGE = "Hello, World!"
REVEAL_DELAY = 0.06      # seconds between typewriter chars
ANIM_SECONDS = 3.0       # how long to animate after reveal
ANIM_FPS = 30            # frames per second

def enable_vt_mode():
    # Enable ANSI colors on Windows 10+ terminals
    if os.name == 'nt':
        try:
            kernel32 = ctypes.windll.kernel32
            h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(h, mode)
        except Exception:
            pass

def rgb_escape(r, g, b, bold=True):
    return f"\033[1m\033[38;2;{r};{g};{b}m" if bold else f"\033[38;2;{r};{g};{b}m"

def reset_escape():
    return "\033[0m"

def hide_cursor(hide=True):
    sys.stdout.write("\033[?25l" if hide else "\033[?25h")
    sys.stdout.flush()

def gradient_text(text, phase):
    # phase is 0..1 – shifts the rainbow across the text
    n = max(1, len(text))
    out = []
    for i, ch in enumerate(text):
        hue = (i / n + phase) % 1.0
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
        out.append(f"{rgb_escape(r, g, b)}{ch}{reset_escape()}")
    return "".join(out)

def center_line(s):
    cols = shutil.get_terminal_size((80, 20)).columns
    pad = max(0, (cols - len(strip_ansi(s))) // 2)
    return " " * pad + s

def strip_ansi(s):
    # lightweight ANSI stripper for length calc; good enough for centering
    import re
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", s)

def main():
    enable_vt_mode()
    hide_cursor(True)
    try:
        # Typewriter reveal with live gradient
        revealed = ""
        t0 = time.time()
        for idx, ch in enumerate(MESSAGE):
            revealed += ch
            phase = ((time.time() - t0) * 0.8) % 1.0
            line = center_line(gradient_text(revealed, phase))
            sys.stdout.write("\r" + " " * shutil.get_terminal_size((80, 20)).columns + "\r")
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(REVEAL_DELAY)

        # Post-reveal shimmer animation
        total_frames = int(ANIM_SECONDS * ANIM_FPS)
        for f in range(total_frames):
            phase = (f / total_frames * 2.0) % 1.0
            # subtle breathing effect: oscillate boldness by mixing with a dim shadow
            colored = gradient_text(MESSAGE, phase)
            line = center_line(colored)
            sys.stdout.write("\r" + " " * shutil.get_terminal_size((80, 20)).columns + "\r")
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(1.0 / ANIM_FPS)

        # Final line with a little sparkle
        sparkle = " ✨"
        final_line = center_line(gradient_text(MESSAGE, 0.25) + sparkle)
        sys.stdout.write("\r" + " " * shutil.get_terminal_size((80, 20)).columns + "\r")
        sys.stdout.write(final_line + reset_escape() + "\n")
        sys.stdout.flush()
    finally:
        hide_cursor(False)
        sys.stdout.write(reset_escape())
        sys.stdout.flush()

if __name__ == "__main__":
    main()