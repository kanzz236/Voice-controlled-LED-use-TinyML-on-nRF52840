import re

with open("model_data.h", "r", encoding="utf-8") as f:
    text = f.read()

match = re.search(r"model_tflite\[\]\s*=\s*\{(.*?)\};", text, re.S)

if not match:
    raise Exception("Không tìm thấy model_tflite[]")

data = bytes(
    int(x, 16)
    for x in re.findall(r"0x([0-9a-fA-F]+)", match.group(1))
)

with open("model.tflite", "wb") as f:
    f.write(data)

print("Đã tạo model.tflite")
print("Size:", len(data), "bytes")