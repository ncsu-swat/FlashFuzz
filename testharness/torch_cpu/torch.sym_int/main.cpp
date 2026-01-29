#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Extract integer values from fuzzer data
        int64_t value1 = 0;
        int64_t value2 = 0;
        std::memcpy(&value1, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
        std::memcpy(&value2, Data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);

        // 1. Create symbolic integers from regular integers
        c10::SymInt sym_int1 = c10::SymInt(value1);
        c10::SymInt sym_int2 = c10::SymInt(value2);

        // 2. Test symbolic integer arithmetic operations
        // Addition
        c10::SymInt result_add = sym_int1 + sym_int2;
        volatile int64_t add_val = result_add.expect_int();

        // Subtraction
        c10::SymInt result_sub = sym_int1 - sym_int2;
        volatile int64_t sub_val = result_sub.expect_int();

        // Multiplication (guard against overflow by using smaller values)
        int32_t small1 = static_cast<int32_t>(value1 & 0xFFFF);
        int32_t small2 = static_cast<int32_t>(value2 & 0xFFFF);
        c10::SymInt sym_small1 = c10::SymInt(small1);
        c10::SymInt sym_small2 = c10::SymInt(small2);
        c10::SymInt result_mul = sym_small1 * sym_small2;
        volatile int64_t mul_val = result_mul.expect_int();

        // Division (avoid division by zero)
        if (value2 != 0) {
            c10::SymInt result_div = sym_int1 / sym_int2;
            volatile int64_t div_val = result_div.expect_int();
        }

        // Modulo (avoid modulo by zero)
        if (value2 != 0) {
            c10::SymInt result_mod = sym_int1 % sym_int2;
            volatile int64_t mod_val = result_mod.expect_int();
        }

        // 3. Test comparison operations
        volatile bool eq = sym_int1 == sym_int2;
        volatile bool neq = sym_int1 != sym_int2;
        volatile bool lt = sym_int1 < sym_int2;
        volatile bool gt = sym_int1 > sym_int2;
        volatile bool lte = sym_int1 <= sym_int2;
        volatile bool gte = sym_int1 >= sym_int2;

        // 4. Test negation
        c10::SymInt neg_result = -sym_int1;
        volatile int64_t neg_val = neg_result.expect_int();

        // 5. Test creating tensor with symbolic-derived shape
        int64_t shape_val = (value1 & 0xFF) + 1;  // 1 to 256
        c10::SymInt sym_shape = c10::SymInt(shape_val);
        torch::Tensor t = torch::zeros({sym_shape.expect_int()});
        volatile int64_t t_size = t.size(0);

        // 6. Test with tensor-derived value
        if (Size > offset + 16) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                if (tensor.numel() == 1) {
                    if (tensor.scalar_type() == torch::kInt ||
                        tensor.scalar_type() == torch::kLong ||
                        tensor.scalar_type() == torch::kShort ||
                        tensor.scalar_type() == torch::kByte) {
                        int64_t tensor_val = tensor.item<int64_t>();
                        c10::SymInt sym_from_tensor = c10::SymInt(tensor_val);
                        volatile int64_t extracted = sym_from_tensor.expect_int();
                    }
                }
            } catch (...) {
                // Inner catch: tensor creation/extraction may fail, that's expected
            }
        }

        // 7. Test guard_int (similar to expect_int but for guarded contexts)
        volatile int64_t guarded = sym_int1.guard_int(__FILE__, __LINE__);

        // 8. Test maybe_as_int
        std::optional<int64_t> maybe_val = sym_int1.maybe_as_int();
        if (maybe_val.has_value()) {
            volatile int64_t concrete = maybe_val.value();
        }

        // 9. Test is_symbolic (should be false for concrete SymInts)
        volatile bool is_sym = sym_int1.is_symbolic();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}