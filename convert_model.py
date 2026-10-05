with open("model.tflite", "rb") as f:
    data = f.read()

with open("model_data.h", "w") as f:
    f.write("#ifndef MODEL_DATA_H_\n#define MODEL_DATA_H_\n\n")
    f.write("#include <cstdint>\n\n")
    f.write("alignas(16) const unsigned char model_tflite[] = {\n")

    for i in range(0, len(data), 12):
        line = ", ".join(f"0x{b:02x}" for b in data[i:i+12])
        f.write("    " + line + ",\n")

    f.write("};\n\n")
    f.write(f"const unsigned int model_tflite_len = {len(data)};\n\n")
    f.write("#endif\n")

print(f"Đã tạo model_data.h mới: {len(data)} bytes")