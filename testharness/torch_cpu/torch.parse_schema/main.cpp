#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful schema string
        if (Size < 4) {
            return 0;
        }
        
        // Create a string from the input data
        std::string schema_str;
        size_t str_length = std::min(Size, static_cast<size_t>(128)); // Limit string length
        
        for (size_t i = 0; i < str_length; i++) {
            // Ensure we have printable characters for the schema
            schema_str += static_cast<char>(32 + (Data[i] % 95)); // ASCII printable range
        }
        
        // Try to parse the schema
        try {
            auto parsed_schema = torch::jit::parseSchema(schema_str);
            
            // Test accessing various parts of the parsed schema
            auto schema_name = parsed_schema.name();
            auto schema_overload_name = parsed_schema.overload_name();
            auto schema_arguments = parsed_schema.arguments();
            auto schema_returns = parsed_schema.returns();
            auto schema_is_vararg = parsed_schema.is_vararg();
            auto schema_is_varret = parsed_schema.is_varret();
            (void)schema_name;
            (void)schema_overload_name;
            (void)schema_arguments;
            (void)schema_returns;
            (void)schema_is_vararg;
            (void)schema_is_varret;
        } catch (const c10::Error& e) {
            // Expected exceptions from invalid schemas
        } catch (const std::runtime_error& e) {
            // Expected exceptions from invalid schemas
        }
        
        // Try with some common schema patterns if we have enough data
        if (Size > 10) {
            std::vector<std::string> common_schemas = {
                "aten::add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
                "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[] stride=1, int[] padding=0, int[] dilation=1, int groups=1) -> Tensor",
                "aten::empty(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
                "aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
                "aten::matmul(Tensor self, Tensor other) -> Tensor",
                "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
                "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
                "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
                "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
                "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"
            };
            
            // Use some bytes from the input to select a schema
            uint8_t schema_idx = Data[offset % Size] % common_schemas.size();
            offset++;
            
            try {
                auto parsed_schema = torch::jit::parseSchema(common_schemas[schema_idx]);
                
                // Test accessing various parts of the parsed schema
                auto schema_name = parsed_schema.name();
                auto schema_overload_name = parsed_schema.overload_name();
                auto schema_arguments = parsed_schema.arguments();
                auto schema_returns = parsed_schema.returns();
                (void)schema_name;
                (void)schema_overload_name;
                (void)schema_arguments;
                (void)schema_returns;
            } catch (const c10::Error& e) {
                // This shouldn't happen with valid schemas, but handle it anyway
            } catch (const std::runtime_error& e) {
                // Handle runtime errors
            }
        }
        
        // Try with malformed schemas if we have enough data
        if (Size > 20) {
            std::vector<std::string> malformed_schemas = {
                "aten:add(Tensor self, Tensor other)", // Missing colon
                "aten::add(Tensor self, Tensor other", // Missing closing parenthesis
                "aten::add(Tensor self, Tensor other) ->", // Missing return type
                "aten::add(Tensor self, Tensor other) -> ", // Empty return type
                "::add(Tensor self, Tensor other) -> Tensor", // Missing namespace
                "aten::add(self, other) -> Tensor", // Missing type annotations
                "aten::add(Tensor self, Tensor other) -> Tensor, Tensor", // Multiple return types without parentheses
                "aten::add(Tensor[] self, Tensor other) -> Tensor", // List type
                "aten::add(Tensor self, Tensor other, *, Scalar alpha=) -> Tensor", // Empty default value
                "aten::add(Tensor self, Tensor other, *, =1) -> Tensor", // Missing parameter name
                "aten::add(Tensor self, Tensor other, *, Scalar alpha=1, ) -> Tensor", // Trailing comma
                "aten::add(Tensor! self, Tensor other) -> Tensor", // Invalid type modifier
                "aten::add(Tensor(a) self, Tensor other) -> Tensor", // Alias annotation
                "aten::add(Tensor self, Tensor other) -> Tensor(a)", // Return alias annotation
                "aten::add(Tensor self, Tensor other) -> Tensor, ", // Trailing comma in return list
                "aten::add(Tensor self, Tensor other) -> (Tensor,)", // Single return in tuple
                "aten::add(Tensor self, Tensor other) -> (Tensor, )", // Trailing comma in return tuple
                "aten::add(Tensor self, Tensor other) -> (Tensor, Tensor", // Unclosed return tuple
                "aten::add(Tensor self, Tensor other) -> Tensor, Tensor)", // Incorrectly closed return list
                "aten::add(Tensor self, Tensor other) -> ()", // Empty return tuple
            };
            
            // Use some bytes from the input to select a malformed schema
            uint8_t schema_idx = Data[offset % Size] % malformed_schemas.size();
            offset++;
            
            try {
                auto parsed_schema = torch::jit::parseSchema(malformed_schemas[schema_idx]);
                
                // If we get here, the schema was actually valid
                auto schema_name = parsed_schema.name();
                auto schema_arguments = parsed_schema.arguments();
                (void)schema_name;
                (void)schema_arguments;
            } catch (const c10::Error& e) {
                // Expected for malformed schemas
            } catch (const std::runtime_error& e) {
                // Expected for malformed schemas
            }
        }
        
        // Try with a schema constructed from input data
        if (Size > 30) {
            // Create components for a schema
            std::string ns = "aten";
            std::string op_name = "custom_op";
            std::string arg_name = "input";
            std::string arg_type = "Tensor";
            
            // Modify components based on input data
            if (offset < Size) {
                uint8_t ns_mod = Data[offset++] % 3;
                if (ns_mod == 1) ns = "prim";
                else if (ns_mod == 2) ns = "custom";
            }
            
            if (offset < Size) {
                uint8_t name_len = 3 + (Data[offset++] % 10); // 3-12 chars
                op_name = "";
                for (uint8_t i = 0; i < name_len && offset < Size; i++) {
                    char c = 'a' + (Data[offset++] % 26);
                    op_name += c;
                }
            }
            
            // Construct a schema string
            std::string custom_schema = ns + "::" + op_name + "(";
            
            // Add arguments
            uint8_t num_args = 1 + (offset < Size ? (Data[offset++] % 5) : 0); // 1-5 args
            for (uint8_t i = 0; i < num_args; i++) {
                if (i > 0) custom_schema += ", ";
                
                // Choose arg type
                std::string this_arg_type = arg_type;
                if (offset < Size) {
                    uint8_t type_mod = Data[offset++] % 7;
                    if (type_mod == 1) this_arg_type = "Scalar";
                    else if (type_mod == 2) this_arg_type = "int";
                    else if (type_mod == 3) this_arg_type = "float";
                    else if (type_mod == 4) this_arg_type = "bool";
                    else if (type_mod == 5) this_arg_type = "str";
                    else if (type_mod == 6) this_arg_type = "Tensor?";
                }
                
                // Add arg name
                std::string this_arg_name = arg_name + std::to_string(i);
                custom_schema += this_arg_type + " " + this_arg_name;
                
                // Maybe add a default value
                if (offset < Size && (Data[offset++] % 4 == 0)) {
                    std::string default_val = "None";
                    if (this_arg_type == "int" || this_arg_type == "float") default_val = "1";
                    else if (this_arg_type == "bool") default_val = "False";
                    else if (this_arg_type == "str") default_val = "\"default\"";
                    custom_schema += "=" + default_val;
                }
            }
            
            custom_schema += ") -> ";
            
            // Add return type(s)
            bool multiple_returns = offset < Size && (Data[offset++] % 3 == 0);
            if (multiple_returns) {
                custom_schema += "(";
                uint8_t num_returns = 2 + (offset < Size ? (Data[offset++] % 3) : 0); // 2-4 returns
                for (uint8_t i = 0; i < num_returns; i++) {
                    if (i > 0) custom_schema += ", ";
                    custom_schema += "Tensor";
                }
                custom_schema += ")";
            } else {
                custom_schema += "Tensor";
            }
            
            try {
                auto parsed_schema = torch::jit::parseSchema(custom_schema);
                
                // Test accessing various parts of the parsed schema
                auto schema_name = parsed_schema.name();
                auto schema_overload_name = parsed_schema.overload_name();
                auto schema_arguments = parsed_schema.arguments();
                auto schema_returns = parsed_schema.returns();
                (void)schema_name;
                (void)schema_overload_name;
                (void)schema_arguments;
                (void)schema_returns;
            } catch (const c10::Error& e) {
                // May happen with invalid schemas
            } catch (const std::runtime_error& e) {
                // May happen with invalid schemas
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}