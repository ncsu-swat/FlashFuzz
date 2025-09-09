#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/public/session.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 2) {
            return 0;
        }
        
        // Extract exit_without_error flag from first byte
        bool exit_without_error = (data[offset] % 2) == 1;
        offset++;
        
        // Extract error message from remaining data
        std::string error_msg;
        if (offset < size) {
            size_t msg_len = std::min(size - offset, static_cast<size_t>(255));
            error_msg = std::string(reinterpret_cast<const char*>(data + offset), msg_len);
            // Ensure null termination and valid UTF-8
            for (char& c : error_msg) {
                if (c == '\0' || static_cast<unsigned char>(c) > 127) {
                    c = 'A';
                }
            }
        }
        
        // Create TensorFlow scope
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        // Create Abort operation
        auto abort_op = tensorflow::ops::Abort(root.WithOpName("test_abort"), 
                                             tensorflow::ops::Abort::ErrorMsg(error_msg)
                                             .ExitWithoutError(exit_without_error));
        
        // Create session and try to run (this should abort/throw)
        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        
        // This should cause an abort or exception
        auto status = session.Run({abort_op}, &outputs);
        
        // If we reach here, something unexpected happened
        if (!status.ok()) {
            std::cout << "Expected abort operation failed with status: " << status.ToString() << std::endl;
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}