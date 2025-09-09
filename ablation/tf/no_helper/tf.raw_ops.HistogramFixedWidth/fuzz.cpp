#include <cstdint>
#include <iostream>
#include <cstring>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/scope.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    try {
        size_t offset = 0;
        
        if (size < 16) return 0;
        
        // Extract number of values (1-100)
        uint32_t num_values = (data[offset] % 100) + 1;
        offset += 1;
        
        // Extract nbins (1-50)
        uint32_t nbins = (data[offset] % 50) + 1;
        offset += 1;
        
        // Extract data type (0: int32, 1: int64, 2: float32, 3: float64)
        uint32_t dtype_idx = data[offset] % 4;
        offset += 1;
        
        // Extract output dtype (0: int32, 1: int64)
        uint32_t output_dtype_idx = data[offset] % 2;
        offset += 1;
        
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        
        if (dtype_idx == 0) { // int32
            // Create values tensor
            std::vector<int32_t> values_data;
            for (uint32_t i = 0; i < num_values && offset + 4 <= size; ++i) {
                int32_t val;
                memcpy(&val, data + offset, sizeof(int32_t));
                values_data.push_back(val);
                offset += 4;
            }
            if (values_data.empty()) return 0;
            
            // Create value_range tensor
            if (offset + 8 > size) return 0;
            int32_t range_min, range_max;
            memcpy(&range_min, data + offset, sizeof(int32_t));
            offset += 4;
            memcpy(&range_max, data + offset, sizeof(int32_t));
            offset += 4;
            
            if (range_min >= range_max) {
                range_max = range_min + 1;
            }
            
            auto values = tensorflow::ops::Const(root, values_data);
            auto value_range = tensorflow::ops::Const(root, {range_min, range_max});
            auto nbins_tensor = tensorflow::ops::Const(root, static_cast<int32_t>(nbins));
            
            tensorflow::DataType output_dtype = (output_dtype_idx == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
            auto hist = tensorflow::ops::HistogramFixedWidth(root, values, value_range, nbins_tensor, 
                                                           tensorflow::ops::HistogramFixedWidth::Dtype(output_dtype));
            
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            session.Run({hist}, &outputs);
            
        } else if (dtype_idx == 1) { // int64
            std::vector<int64_t> values_data;
            for (uint32_t i = 0; i < num_values && offset + 8 <= size; ++i) {
                int64_t val;
                memcpy(&val, data + offset, sizeof(int64_t));
                values_data.push_back(val);
                offset += 8;
            }
            if (values_data.empty()) return 0;
            
            if (offset + 16 > size) return 0;
            int64_t range_min, range_max;
            memcpy(&range_min, data + offset, sizeof(int64_t));
            offset += 8;
            memcpy(&range_max, data + offset, sizeof(int64_t));
            offset += 8;
            
            if (range_min >= range_max) {
                range_max = range_min + 1;
            }
            
            auto values = tensorflow::ops::Const(root, values_data);
            auto value_range = tensorflow::ops::Const(root, {range_min, range_max});
            auto nbins_tensor = tensorflow::ops::Const(root, static_cast<int32_t>(nbins));
            
            tensorflow::DataType output_dtype = (output_dtype_idx == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
            auto hist = tensorflow::ops::HistogramFixedWidth(root, values, value_range, nbins_tensor,
                                                           tensorflow::ops::HistogramFixedWidth::Dtype(output_dtype));
            
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            session.Run({hist}, &outputs);
            
        } else if (dtype_idx == 2) { // float32
            std::vector<float> values_data;
            for (uint32_t i = 0; i < num_values && offset + 4 <= size; ++i) {
                float val;
                memcpy(&val, data + offset, sizeof(float));
                if (!std::isfinite(val)) val = 0.0f;
                values_data.push_back(val);
                offset += 4;
            }
            if (values_data.empty()) return 0;
            
            if (offset + 8 > size) return 0;
            float range_min, range_max;
            memcpy(&range_min, data + offset, sizeof(float));
            offset += 4;
            memcpy(&range_max, data + offset, sizeof(float));
            offset += 4;
            
            if (!std::isfinite(range_min)) range_min = 0.0f;
            if (!std::isfinite(range_max)) range_max = 1.0f;
            if (range_min >= range_max) {
                range_max = range_min + 1.0f;
            }
            
            auto values = tensorflow::ops::Const(root, values_data);
            auto value_range = tensorflow::ops::Const(root, {range_min, range_max});
            auto nbins_tensor = tensorflow::ops::Const(root, static_cast<int32_t>(nbins));
            
            tensorflow::DataType output_dtype = (output_dtype_idx == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
            auto hist = tensorflow::ops::HistogramFixedWidth(root, values, value_range, nbins_tensor,
                                                           tensorflow::ops::HistogramFixedWidth::Dtype(output_dtype));
            
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            session.Run({hist}, &outputs);
            
        } else { // float64
            std::vector<double> values_data;
            for (uint32_t i = 0; i < num_values && offset + 8 <= size; ++i) {
                double val;
                memcpy(&val, data + offset, sizeof(double));
                if (!std::isfinite(val)) val = 0.0;
                values_data.push_back(val);
                offset += 8;
            }
            if (values_data.empty()) return 0;
            
            if (offset + 16 > size) return 0;
            double range_min, range_max;
            memcpy(&range_min, data + offset, sizeof(double));
            offset += 8;
            memcpy(&range_max, data + offset, sizeof(double));
            offset += 8;
            
            if (!std::isfinite(range_min)) range_min = 0.0;
            if (!std::isfinite(range_max)) range_max = 1.0;
            if (range_min >= range_max) {
                range_max = range_min + 1.0;
            }
            
            auto values = tensorflow::ops::Const(root, values_data);
            auto value_range = tensorflow::ops::Const(root, {range_min, range_max});
            auto nbins_tensor = tensorflow::ops::Const(root, static_cast<int32_t>(nbins));
            
            tensorflow::DataType output_dtype = (output_dtype_idx == 0) ? tensorflow::DT_INT32 : tensorflow::DT_INT64;
            auto hist = tensorflow::ops::HistogramFixedWidth(root, values, value_range, nbins_tensor,
                                                           tensorflow::ops::HistogramFixedWidth::Dtype(output_dtype));
            
            tensorflow::ClientSession session(root);
            std::vector<tensorflow::Tensor> outputs;
            session.Run({hist}, &outputs);
        }
        
    } catch (const std::exception& e) {
        // print Exception to stderr, do not remove this
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}