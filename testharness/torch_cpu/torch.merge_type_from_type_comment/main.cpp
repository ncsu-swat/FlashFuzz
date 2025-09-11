#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a string from the remaining data to use as type comment
        std::string type_comment;
        if (offset < Size) {
            size_t comment_length = std::min(Size - offset, static_cast<size_t>(32));
            type_comment.reserve(comment_length);
            for (size_t i = 0; i < comment_length; ++i) {
                type_comment.push_back(static_cast<char>(Data[offset + i]));
            }
            offset += comment_length;
        } else {
            type_comment = "Tensor";
        }
        
        // Create a variable name
        std::string var_name;
        if (offset < Size) {
            size_t name_length = std::min(Size - offset, static_cast<size_t>(16));
            var_name.reserve(name_length);
            for (size_t i = 0; i < name_length; ++i) {
                var_name.push_back(static_cast<char>(Data[offset + i]));
            }
            offset += name_length;
        } else {
            var_name = "x";
        }
        
        // Create a source type
        c10::TypePtr source_type = c10::TensorType::create(tensor);
        
        // Create a type annotation
        c10::TypePtr type_annotation;
        if (offset < Size && Data[offset] % 2 == 0) {
            // Sometimes use a different tensor type
            std::vector<int64_t> sizes;
            uint8_t rank = Data[offset] % 5;
            offset++;
            
            for (uint8_t i = 0; i < rank && offset < Size; i++) {
                sizes.push_back(static_cast<int64_t>(Data[offset++]) % 10);
            }
            
            c10::VaryingShape<int64_t> shape(sizes);
            c10::VaryingShape<int64_t> strides;
            
            type_annotation = c10::TensorType::create(
                tensor.scalar_type(),
                c10::kCPU,
                shape,
                strides
            );
        } else {
            // Sometimes use the same type
            type_annotation = source_type;
        }
        
        // Call the merge_type_from_type_comment function
        c10::TypePtr merged_type;
        try {
            merged_type = torch::jit::merge_type_from_type_comment(source_type, type_annotation, var_name, type_comment);
            
            // Use the merged type to avoid optimization
            if (merged_type) {
                auto type_kind = merged_type->kind();
                auto type_str = merged_type->str();
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and handled
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
