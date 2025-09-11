#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#define MAX_RANK 4
#define MIN_RANK 0
#define MIN_TENSOR_SHAPE_DIMS_TF 1
#define MAX_TENSOR_SHAPE_DIMS_TF 10
#define MAX_NUM_TENSORS 5

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* data, size_t size) {
    std::cerr << "Error: " << message << std::endl;
}
}

tensorflow::DataType parseDataType(uint8_t selector) {
    tensorflow::DataType dtype; 
    switch (selector % 12) {  
        case 0:
            dtype = tensorflow::DT_FLOAT;
            break;
        case 1:
            dtype = tensorflow::DT_DOUBLE;
            break;
        case 2:
            dtype = tensorflow::DT_INT32;
            break;
        case 3:
            dtype = tensorflow::DT_UINT8;
            break;
        case 4:
            dtype = tensorflow::DT_INT16;
            break;
        case 5:
            dtype = tensorflow::DT_INT8;
            break;
        case 6:
            dtype = tensorflow::DT_INT64;
            break;
        case 7:
            dtype = tensorflow::DT_BOOL;
            break;
        case 8:
            dtype = tensorflow::DT_UINT16;
            break;
        case 9:
            dtype = tensorflow::DT_UINT32;
            break;
        case 10:
            dtype = tensorflow::DT_UINT64;
            break;
        case 11:
            dtype = tensorflow::DT_BFLOAT16;
            break;
    }
    return dtype;
}

uint8_t parseRank(uint8_t byte) {
    constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
    uint8_t rank = byte % range + MIN_RANK;
    return rank;
}

std::vector<int64_t> parseShape(const uint8_t* data, size_t& offset, size_t total_size, uint8_t rank) {
    if (rank == 0) {
        return {};
    }

    std::vector<int64_t> shape;
    shape.reserve(rank);
    const auto sizeof_dim = sizeof(int64_t);

    for (uint8_t i = 0; i < rank; ++i) {
        if (offset + sizeof_dim <= total_size) {
            int64_t dim_val;
            std::memcpy(&dim_val, data + offset, sizeof_dim);
            offset += sizeof_dim;
            
            dim_val = MIN_TENSOR_SHAPE_DIMS_TF +
                    static_cast<int64_t>((static_cast<uint64_t>(std::abs(dim_val)) %
                                        static_cast<uint64_t>(MAX_TENSOR_SHAPE_DIMS_TF - MIN_TENSOR_SHAPE_DIMS_TF + 1)));

            shape.push_back(dim_val);
        } else {
             shape.push_back(1);
        }
    }

    return shape;
}

template <typename T>
void fillTensorWithData(tensorflow::Tensor& tensor, const uint8_t* data,
                        size_t& offset, size_t total_size) {
    auto flat = tensor.flat<T>();
    const size_t num_elements = flat.size();
    const size_t element_size = sizeof(T);

    for (size_t i = 0; i < num_elements; ++i) {
        if (offset + element_size <= total_size) {
            T value;
            std::memcpy(&value, data + offset, element_size);
            offset += element_size;
            flat(i) = value;
        } else {
            flat(i) = T{};
        }
    }
}

void fillTensorWithDataByType(tensorflow::Tensor& tensor,
                              tensorflow::DataType dtype, const uint8_t* data,
                              size_t& offset, size_t total_size) {
    switch (dtype) {
        case tensorflow::DT_FLOAT:
            fillTensorWithData<float>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_DOUBLE:
            fillTensorWithData<double>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT32:
            fillTensorWithData<int32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT8:
            fillTensorWithData<uint8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT16:
            fillTensorWithData<int16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT8:
            fillTensorWithData<int8_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_INT64:
            fillTensorWithData<int64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BOOL:
            fillTensorWithData<bool>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT16:
            fillTensorWithData<uint16_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT32:
            fillTensorWithData<uint32_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_UINT64:
            fillTensorWithData<uint64_t>(tensor, data, offset, total_size);
            break;
        case tensorflow::DT_BFLOAT16:
            fillTensorWithData<tensorflow::bfloat16>(tensor, data, offset, total_size);
            break;
        default:
            break;
    }
}

std::string generateFilename(const uint8_t* data, size_t& offset, size_t total_size) {
    std::stringstream ss;
    ss << "/tmp/test_save_slices_";
    
    for (int i = 0; i < 8 && offset < total_size; ++i) {
        ss << static_cast<int>(data[offset]);
        offset++;
    }
    
    return ss.str();
}

std::string generateTensorName(int index, const uint8_t* data, size_t& offset, size_t total_size) {
    std::stringstream ss;
    ss << "tensor_" << index;
    
    if (offset < total_size) {
        ss << "_" << static_cast<int>(data[offset]);
        offset++;
    }
    
    return ss.str();
}

std::string generateShapeAndSlice(const std::vector<int64_t>& shape, const uint8_t* data, size_t& offset, size_t total_size) {
    if (offset >= total_size) {
        return "";
    }
    
    uint8_t choice = data[offset] % 3;
    offset++;
    
    if (choice == 0) {
        return "";
    }
    
    std::stringstream ss;
    
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << " ";
        ss << shape[i];
    }
    
    if (choice == 1) {
        ss << " ";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ":";
            ss << "-";
        }
    } else {
        ss << " ";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ":";
            int64_t start = 0;
            int64_t length = shape[i];
            if (offset + 1 < total_size) {
                start = data[offset] % shape[i];
                length = 1 + (data[offset + 1] % (shape[i] - start));
                offset += 2;
            }
            ss << start << "," << length;
        }
    }
    
    return ss.str();
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::cout << "Start Fuzzing" << std::endl;
    if (size < 10) return 0;
    
    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        std::string filename = generateFilename(data, offset, size);
        
        uint8_t num_tensors = 1 + (data[offset] % MAX_NUM_TENSORS);
        offset++;
        
        std::vector<std::string> tensor_names;
        std::vector<std::string> shapes_and_slices;
        std::vector<tensorflow::Tensor> data_tensors;
        std::vector<tensorflow::DataType> data_types;
        
        for (uint8_t i = 0; i < num_tensors && offset < size; ++i) {
            if (offset + 3 >= size) break;
            
            tensorflow::DataType dtype = parseDataType(data[offset]);
            offset++;
            
            uint8_t rank = parseRank(data[offset]);
            offset++;
            
            std::vector<int64_t> shape = parseShape(data, offset, size, rank);
            
            tensorflow::TensorShape tensor_shape;
            for (int64_t dim : shape) {
                tensor_shape.AddDim(dim);
            }
            
            tensorflow::Tensor tensor(dtype, tensor_shape);
            fillTensorWithDataByType(tensor, dtype, data, offset, size);
            
            std::string tensor_name = generateTensorName(i, data, offset, size);
            std::string shape_and_slice = generateShapeAndSlice(shape, data, offset, size);
            
            tensor_names.push_back(tensor_name);
            shapes_and_slices.push_back(shape_and_slice);
            data_tensors.push_back(tensor);
            data_types.push_back(dtype);
        }
        
        if (tensor_names.empty()) return 0;
        
        tensorflow::Tensor filename_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        filename_tensor.scalar<tensorflow::tstring>()() = filename;
        
        tensorflow::Tensor tensor_names_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({static_cast<int64_t>(tensor_names.size())}));
        auto tensor_names_flat = tensor_names_tensor.flat<tensorflow::tstring>();
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            tensor_names_flat(i) = tensor_names[i];
        }
        
        tensorflow::Tensor shapes_and_slices_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({static_cast<int64_t>(shapes_and_slices.size())}));
        auto shapes_and_slices_flat = shapes_and_slices_tensor.flat<tensorflow::tstring>();
        for (size_t i = 0; i < shapes_and_slices.size(); ++i) {
            shapes_and_slices_flat(i) = shapes_and_slices[i];
        }
        
        auto filename_input = tensorflow::ops::Const(root, filename_tensor);
        auto tensor_names_input = tensorflow::ops::Const(root, tensor_names_tensor);
        auto shapes_and_slices_input = tensorflow::ops::Const(root, shapes_and_slices_tensor);
        
        // Convert std::vector<tensorflow::Input> to tensorflow::InputList
        tensorflow::InputList data_inputs;
        for (const auto& tensor : data_tensors) {
            data_inputs.push_back(tensorflow::ops::Const(root, tensor));
        }
        
        auto save_slices_op = tensorflow::ops::SaveSlices(
            root,
            filename_input,
            tensor_names_input,
            shapes_and_slices_input,
            data_inputs
        );
        
        tensorflow::ClientSession session(root);
        tensorflow::Status status = session.Run({save_slices_op}, nullptr);
        if (!status.ok()) {
            return -1;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    } 

    return 0;
}
