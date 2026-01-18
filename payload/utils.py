# payload/utils.py

def text_to_bits(text):
    """
    Convert string to list of bits (0/1).
    Each character -> 8 bits (ASCII).
    """
    bits = []
    for char in text:
        bin_val = format(ord(char), '08b')
        bits.extend([int(b) for b in bin_val])
    return bits


def bits_to_text(bits):
    """
    Convert list of bits back to string.
    """
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)


def truncate_bits(bits, max_len):
    """
    Safety helper: trims payload if capacity is smaller.
    """
    return bits[:max_len]
