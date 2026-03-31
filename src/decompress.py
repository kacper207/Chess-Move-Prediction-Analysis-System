import zstandard as zstd
import os

input_path = r"C:\Users\HARDPC\PycharmProjects\chess_predictor\data\raw\lichess01.pgn.zst"
output_path = r"C:\Users\HARDPC\PycharmProjects\chess_predictor\data\raw\lichess01.pgn"

compressed_size = os.path.getsize(input_path)

print("Rozpakowywanie...")
with open(input_path, "rb") as compressed:
    dctx = zstd.ZstdDecompressor()
    with open(output_path, "wb") as output:
        bytes_read = 0
        reader = dctx.stream_reader(compressed)
        while True:
            chunk = reader.read(8 * 1024 * 1024)  # 8 MB na raz
            if not chunk:
                break
            output.write(chunk)
            bytes_read += len(chunk)
            percent = (compressed.tell() / compressed_size) * 100
            print(f"\r  Postęp: {percent:.1f}%  ({compressed.tell() / 1024**3:.2f} GB / {compressed_size / 1024**3:.2f} GB)", end="")

print("\nGotowe!")