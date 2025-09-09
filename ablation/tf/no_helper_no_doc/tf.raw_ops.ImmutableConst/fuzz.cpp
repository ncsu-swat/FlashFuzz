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
        
        // Extract dtype from fuzzer input
        uint8_t dtype_val = data[offset++] % 19; // TensorFlow has ~19 basic dtypes
        tensorflow::DataType dtype;
        switch (dtype_val) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_UINT8; break;
            case 4: dtype = tensorflow::DT_INT16; break;
            case 5: dtype = tensorflow::DT_INT8; break;
            case 6: dtype = tensorflow::DT_STRING; break;
            case 7: dtype = tensorflow::DT_INT64; break;
            case 8: dtype = tensorflow::DT_BOOL; break;
            case 9: dtype = tensorflow::DT_QINT8; break;
            case 10: dtype = tensorflow::DT_QUINT8; break;
            case 11: dtype = tensorflow::DT_QINT32; break;
            case 12: dtype = tensorflow::DT_BFLOAT16; break;
            case 13: dtype = tensorflow::DT_QINT16; break;
            case 14: dtype = tensorflow::DT_QUINT16; break;
            case 15: dtype = tensorflow::DT_UINT16; break;
            case 16: dtype = tensorflow::DT_UINT32; break;
            case 17: dtype = tensorflow::DT_UINT64; break;
            default: dtype = tensorflow::DT_FLOAT; break;
        }
        
        // Extract shape dimensions
        if (offset >= size) return 0;
        uint8_t num_dims = (data[offset++] % 4) + 1; // 1-4 dimensions
        
        std::vector<int64_t> shape_dims;
        for (int i = 0; i < num_dims && offset + 1 < size; i++) {
            int64_t dim = (data[offset] | (data[offset + 1] << 8)) % 10 + 1; // 1-10 size per dim
            shape_dims.push_back(dim);
            offset += 2;
        }
        
        if (shape_dims.empty()) {
            shape_dims.push_back(1);
        }
        
        tensorflow::TensorShape shape(shape_dims);
        
        // Create memory region path (simulate file path)
        std::string memory_region_name = "/tmp/test_region_";
        if (offset < size) {
            memory_region_name += std::to_string(data[offset++] % 1000);
        } else {
            memory_region_name += "0";
        }
        
        // Create a simple test using NodeDefBuilder
        tensorflow::NodeDef node_def;
        tensorflow::NodeDefBuilder builder("test_immutable_const", "ImmutableConst");
        builder.Attr("dtype", dtype);
        builder.Attr("shape", shape);
        builder.Attr("memory_region_name", memory_region_name);
        
        auto status = builder.Finalize(&node_def);
        if (!status.ok()) {
            return 0;
        }
        
        // Create OpKernel context for testing
        tensorflow::OpKernelContext::Params params;
        std::unique_ptr<tensorflow::OpKernel> kernel;
        
        // Try to create the kernel
        status = tensorflow::CreateOpKernel(tensorflow::DEVICE_CPU, nullptr, 
                                          tensorflow::NodeDef(node_def), 
                                          tensorflow::OpKernelContext::kGraphMode, 
                                          &kernel);
        
        if (status.ok() && kernel) {
            // Basic validation that kernel was created successfully
            // In a real scenario, we would need proper context setup to actually run the kernel
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}