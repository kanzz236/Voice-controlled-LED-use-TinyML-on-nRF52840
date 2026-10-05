
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "../model_data.h"

constexpr int kTensorArenaSize = 120 * 1024;

alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

int main()
{
    printk("\r\n=== nRF52840 TFLM TEST ===\r\n");

    const tflite::Model *model = tflite::GetModel(model_tflite);

    printk("Model version: %d\r\n", model->version());
    printk("TFLite schema: %d\r\n", TFLITE_SCHEMA_VERSION);
    printk("Subgraphs: %d\r\n", model->subgraphs()->size());

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printk("ERROR: Model schema mismatch\r\n");
        return 0;
    }

    static tflite::MicroMutableOpResolver<12> resolver;

resolver.AddConv2D();
resolver.AddMul();
resolver.AddAdd();
resolver.AddMaxPool2D();
resolver.AddShape();
resolver.AddStridedSlice();
resolver.AddPack();
resolver.AddReshape();
resolver.AddFullyConnected();
resolver.AddSoftmax();

    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    printk("Arena size: %d bytes\r\n", kTensorArenaSize);

    TfLiteStatus status = interpreter.AllocateTensors();

    if (status != kTfLiteOk) {
        printk("ERROR: AllocateTensors failed\r\n");
        return 0;
    }

    printk("AllocateTensors OK!\r\n");

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    printk("Input type: %d\r\n", input->type);
    printk("Output type: %d\r\n", output->type);
    printk("Input bytes: %d\r\n", input->bytes);
    printk("Output bytes: %d\r\n", output->bytes);

    printk("Input dimensions: ");
    for (unsigned int i = 0; i < input->dims->size; i++) {
        printk("%d ", input->dims->data[i]);
    }
    printk("\r\n");

    printk("Output dimensions: ");
    for (unsigned int i = 0; i < output->dims->size; i++) {
        printk("%d ", output->dims->data[i]);
    }
    printk("\r\n");

   for (size_t i = 0; i < input->bytes / sizeof(float); i++) {
    input->data.f[i] = 0.0f;
}

    printk("Running inference...\r\n");

    if (interpreter.Invoke() != kTfLiteOk) {
        printk("ERROR: Invoke failed\r\n");
        return 0;
    }

    printk("Inference OK!\r\n");

 for (int i = 0; i < 4; i++) {
    int value = (int)(output->data.f[i] * 1000.0f);
    printk("output[%d] = %d / 1000\r\n", i, value);
}
    while (1) {
        k_sleep(K_SECONDS(1));
    }

    return 0;
}
