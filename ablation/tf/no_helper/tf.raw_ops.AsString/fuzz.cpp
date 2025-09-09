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
        
        if (size < 20) return 0;
        
        // Extract parameters from fuzzer input
        int precision = static_cast<int>(data[offset]) - 128;
        offset++;
        
        bool scientific = data[offset] % 2 == 1;
        offset++;
        
        bool shortest = data[offset] % 2 == 1;
        offset++;
        
        int width = static_cast<int>(data[offset]) - 128;
        offset++;
        
        char fill_char = static_cast<char>(data[offset] % 128);
        if (fill_char == 0) fill_char = ' ';
        std::string fill(1, fill_char);
        offset++;
        
        // Determine tensor type and shape
        int type_idx = data[offset] % 16;
        offset++;
        
        tensorflow::DataType dtype;
        switch (type_idx) {
            case 0: dtype = tensorflow::DT_FLOAT; break;
            case 1: dtype = tensorflow::DT_DOUBLE; break;
            case 2: dtype = tensorflow::DT_INT32; break;
            case 3: dtype = tensorflow::DT_UINT8; break;
            case 4: dtype = tensorflow::DT_INT16; break;
            case 5: dtype = tensorflow::DT_INT8; break;
            case 6: dtype = tensorflow::DT_INT64; break;
            case 7: dtype = tensorflow::DT_BFLOAT16; break;
            case 8: dtype = tensorflow::DT_UINT16; break;
            case 9: dtype = tensorflow::DT_HALF; break;
            case 10: dtype = tensorflow::DT_UINT32; break;
            case 11: dtype = tensorflow::DT_UINT64; break;
            case 12: dtype = tensorflow::DT_COMPLEX64; break;
            case 13: dtype = tensorflow::DT_COMPLEX128; break;
            case 14: dtype = tensorflow::DT_BOOL; break;
            default: dtype = tensorflow::DT_STRING; break;
        }
        
        // Create tensor shape
        int num_dims = (data[offset] % 4) + 1;
        offset++;
        
        tensorflow::TensorShape shape;
        for (int i = 0; i < num_dims && offset < size; i++) {
            int dim_size = (data[offset] % 10) + 1;
            shape.AddDim(dim_size);
            offset++;
        }
        
        // Create input tensor
        tensorflow::Tensor input_tensor(dtype, shape);
        
        // Fill tensor with data
        int64_t num_elements = shape.num_elements();
        if (num_elements > 1000) num_elements = 1000; // Limit for fuzzing
        
        switch (dtype) {
            case tensorflow::DT_FLOAT: {
                auto flat = input_tensor.flat<float>();
                for (int64_t i = 0; i < num_elements && offset + 4 <= size; i++) {
                    float val;
                    memcpy(&val, data + offset, sizeof(float));
                    flat(i) = val;
                    offset += 4;
                }
                break;
            }
            case tensorflow::DT_DOUBLE: {
                auto flat = input_tensor.flat<double>();
                for (int64_t i = 0; i < num_elements && offset + 8 <= size; i++) {
                    double val;
                    memcpy(&val, data + offset, sizeof(double));
                    flat(i) = val;
                    offset += 8;
                }
                break;
            }
            case tensorflow::DT_INT32: {
                auto flat = input_tensor.flat<int32_t>();
                for (int64_t i = 0; i < num_elements && offset + 4 <= size; i++) {
                    int32_t val;
                    memcpy(&val, data + offset, sizeof(int32_t));
                    flat(i) = val;
                    offset += 4;
                }
                break;
            }
            case tensorflow::DT_INT64: {
                auto flat = input_tensor.flat<int64_t>();
                for (int64_t i = 0; i < num_elements && offset + 8 <= size; i++) {
                    int64_t val;
                    memcpy(&val, data + offset, sizeof(int64_t));
                    flat(i) = val;
                    offset += 8;
                }
                break;
            }
            case tensorflow::DT_BOOL: {
                auto flat = input_tensor.flat<bool>();
                for (int64_t i = 0; i < num_elements && offset < size; i++) {
                    flat(i) = data[offset] % 2 == 1;
                    offset++;
                }
                break;
            }
            default:
                // For other types, just use int32 data
                if (dtype == tensorflow::DT_UINT8) {
                    auto flat = input_tensor.flat<uint8_t>();
                    for (int64_t i = 0; i < num_elements && offset < size; i++) {
                        flat(i) = data[offset];
                        offset++;
                    }
                }
                break;
        }
        
        // Create session and graph
        tensorflow::GraphDef graph_def;
        tensorflow::NodeDef* node_def = graph_def.add_node();
        
        node_def->set_name("as_string_op");
        node_def->set_op("AsString");
        node_def->add_input("input:0");
        
        // Set attributes
        tensorflow::AttrValue precision_attr;
        precision_attr.set_i(precision);
        (*node_def->mutable_attr())["precision"] = precision_attr;
        
        tensorflow::AttrValue scientific_attr;
        scientific_attr.set_b(scientific);
        (*node_def->mutable_attr())["scientific"] = scientific_attr;
        
        tensorflow::AttrValue shortest_attr;
        shortest_attr.set_b(shortest);
        (*node_def->mutable_attr())["shortest"] = shortest_attr;
        
        tensorflow::AttrValue width_attr;
        width_attr.set_i(width);
        (*node_def->mutable_attr())["width"] = width_attr;
        
        tensorflow::AttrValue fill_attr;
        fill_attr.set_s(fill);
        (*node_def->mutable_attr())["fill"] = fill_attr;
        
        // Add input node
        tensorflow::NodeDef* input_node = graph_def.add_node();
        input_node->set_name("input");
        input_node->set_op("Placeholder");
        tensorflow::AttrValue dtype_attr;
        dtype_attr.set_type(dtype);
        (*input_node->mutable_attr())["dtype"] = dtype_attr;
        
        // Create session
        tensorflow::SessionOptions options;
        std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
        
        if (session) {
            tensorflow::Status status = session->Create(graph_def);
            if (status.ok()) {
                // Run the operation
                std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
                    {"input:0", input_tensor}
                };
                
                std::vector<tensorflow::Tensor> outputs;
                status = session->Run(inputs, {"as_string_op:0"}, {}, &outputs);
                
                if (status.ok() && !outputs.empty()) {
                    // Successfully executed AsString operation
                    auto output_flat = outputs[0].flat<tensorflow::tstring>();
                    // Access first few elements to ensure they're valid
                    for (int i = 0; i < std::min(static_cast<int64_t>(5), outputs[0].NumElements()); i++) {
                        std::string str_val = std::string(output_flat(i));
                        // Just access the string to ensure it's valid
                        if (!str_val.empty()) {
                            volatile char c = str_val[0];
                            (void)c;
                        }
                    }
                }
            }
            session->Close();
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}