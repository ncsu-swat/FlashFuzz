#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/framework/node_def_builder.h>
#include <tensorflow/core/kernels/ops_util.h>
#include <tensorflow/core/common_runtime/kernel_benchmark_testlib.h>
#include <tensorflow/core/platform/test.h>
#include <tensorflow/core/framework/fake_input.h>
#include <tensorflow/core/kernels/ops_testutil.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 8) return 0;
        
        // Extract tensor dimensions
        uint32_t num_dims = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += sizeof(uint32_t);
        num_dims = num_dims % 5; // Limit dimensions to reasonable size
        
        if (offset + num_dims * sizeof(uint32_t) > size) return 0;
        
        // Extract tensor shape
        tensorflow::TensorShape shape;
        for (uint32_t i = 0; i < num_dims; ++i) {
            uint32_t dim_size = *reinterpret_cast<const uint32_t*>(data + offset);
            offset += sizeof(uint32_t);
            dim_size = (dim_size % 100) + 1; // Limit dimension size
            shape.AddDim(dim_size);
        }
        
        if (offset >= size) return 0;
        
        // Extract data type
        uint8_t dtype_val = data[offset++] % 19; // Limit to valid TensorFlow data types
        tensorflow::DataType dtype;
        
        switch (dtype_val) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_UINT8; break;
            case 4: dtype = tensorflow::DT_INT16; break;
            case 5: dtype = tensorflow::DT_INT8; break;
            case 6: dtype = tensorflow::DT_STRING; break;
            case 7: dtype = tensorflow::DT_COMPLEX64; break;
            case 8: dtype = tensorflow::DT_INT64; break;
            case 9: dtype = tensorflow::DT_BOOL; break;
            case 10: dtype = tensorflow::DT_QINT8; break;
            case 11: dtype = tensorflow::DT_QUINT8; break;
            case 12: dtype = tensorflow::DT_QINT32; break;
            case 13: dtype = tensorflow::DT_BFLOAT16; break;
            case 14: dtype = tensorflow::DT_QINT16; break;
            case 15: dtype = tensorflow::DT_QUINT16; break;
            case 16: dtype = tensorflow::DT_UINT16; break;
            case 17: dtype = tensorflow::DT_COMPLEX128; break;
            case 18: dtype = tensorflow::DT_HALF; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Create tensor
        tensorflow::Tensor tensor(dtype, shape);
        
        // Fill tensor with remaining data
        size_t tensor_bytes = tensor.TotalBytes();
        if (tensor_bytes > 0 && offset < size) {
            size_t available_bytes = size - offset;
            size_t copy_bytes = std::min(tensor_bytes, available_bytes);
            
            if (dtype == tensorflow::DT_STRING) {
                // Handle string tensors specially
                auto flat = tensor.flat<tensorflow::tstring>();
                for (int64_t i = 0; i < flat.size() && offset < size; ++i) {
                    size_t str_len = std::min(static_cast<size_t>(32), size - offset);
                    flat(i) = tensorflow::tstring(reinterpret_cast<const char*>(data + offset), str_len);
                    offset += str_len;
                }
            } else {
                std::memcpy(tensor.data(), data + offset, copy_bytes);
            }
        }
        
        // Create OpKernelContext for SerializeTensor
        tensorflow::NodeDef node_def;
        node_def.set_name("serialize_tensor_test");
        node_def.set_op("SerializeTensor");
        
        tensorflow::OpKernelConstruction construction(
            tensorflow::DeviceType("CPU"), nullptr, &node_def, nullptr, nullptr);
        
        // Create a simple test using OpsTestBase
        class SerializeTensorTest : public tensorflow::OpsTestBase {
        public:
            void RunTest(const tensorflow::Tensor& input_tensor) {
                TF_ASSERT_OK(NodeDefBuilder("serialize_tensor", "SerializeTensor")
                    .Input(FakeInput(input_tensor.dtype()))
                    .Finalize(node_def()));
                TF_ASSERT_OK(InitOp());
                AddInputFromArray<tensorflow::Tensor>(input_tensor.shape(), {input_tensor});
                TF_ASSERT_OK(RunOpKernel());
                
                // Get output
                tensorflow::Tensor* output_tensor = GetOutput(0);
                if (output_tensor != nullptr) {
                    // Verify output is a string tensor
                    if (output_tensor->dtype() == tensorflow::DT_STRING) {
                        auto output_flat = output_tensor->flat<tensorflow::tstring>();
                        // Basic validation that serialization produced some output
                        if (output_flat.size() > 0 && !output_flat(0).empty()) {
                            // Serialization successful
                        }
                    }
                }
            }
        };
        
        SerializeTensorTest test;
        test.RunTest(tensor);
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}