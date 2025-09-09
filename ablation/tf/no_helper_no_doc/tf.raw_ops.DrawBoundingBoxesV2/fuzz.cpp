#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract dimensions from fuzz data
        int batch_size = (data[offset] % 4) + 1;
        offset++;
        int height = (data[offset] % 128) + 32;
        offset++;
        int width = (data[offset] % 128) + 32;
        offset++;
        int channels = (data[offset] % 3) + 1;
        offset++;
        int num_boxes = (data[offset] % 10) + 1;
        offset++;
        int num_colors = (data[offset] % 10) + 1;
        offset++;
        
        if (offset + 10 > size) return 0;
        
        // Create images tensor [batch_size, height, width, channels]
        tensorflow::Tensor images(tensorflow::DT_FLOAT, 
            tensorflow::TensorShape({batch_size, height, width, channels}));
        auto images_flat = images.flat<float>();
        for (int i = 0; i < images_flat.size() && offset < size; i++) {
            images_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create boxes tensor [batch_size, num_boxes, 4]
        tensorflow::Tensor boxes(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({batch_size, num_boxes, 4}));
        auto boxes_flat = boxes.flat<float>();
        for (int i = 0; i < boxes_flat.size() && offset < size; i++) {
            boxes_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create colors tensor [num_colors, 4] (RGBA)
        tensorflow::Tensor colors(tensorflow::DT_FLOAT,
            tensorflow::TensorShape({num_colors, 4}));
        auto colors_flat = colors.flat<float>();
        for (int i = 0; i < colors_flat.size() && offset < size; i++) {
            colors_flat(i) = static_cast<float>(data[offset % size]) / 255.0f;
            offset++;
        }
        
        // Create a simple test using OpsTestBase
        class DrawBoundingBoxesV2Test : public tensorflow::OpsTestBase {
        public:
            void RunTest(const tensorflow::Tensor& images_input,
                        const tensorflow::Tensor& boxes_input,
                        const tensorflow::Tensor& colors_input) {
                tensorflow::NodeDefBuilder builder("draw_bounding_boxes_v2", "DrawBoundingBoxesV2");
                tensorflow::NodeDef node_def;
                TF_CHECK_OK(builder.Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                                  .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                                  .Input(tensorflow::FakeInput(tensorflow::DT_FLOAT))
                                  .Finalize(&node_def));
                
                TF_CHECK_OK(InitOp(node_def));
                
                AddInputFromArray<float>(images_input.shape(), images_input.flat<float>());
                AddInputFromArray<float>(boxes_input.shape(), boxes_input.flat<float>());
                AddInputFromArray<float>(colors_input.shape(), colors_input.flat<float>());
                
                TF_CHECK_OK(RunOpKernel());
                
                // Get output tensor
                tensorflow::Tensor* output = GetOutput(0);
                if (output != nullptr) {
                    // Verify output shape matches input images shape
                    if (output->shape().dims() == images_input.shape().dims()) {
                        for (int i = 0; i < output->shape().dims(); i++) {
                            if (output->shape().dim_size(i) != images_input.shape().dim_size(i)) {
                                return;
                            }
                        }
                    }
                }
            }
        };
        
        DrawBoundingBoxesV2Test test;
        test.RunTest(images, boxes, colors);
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}