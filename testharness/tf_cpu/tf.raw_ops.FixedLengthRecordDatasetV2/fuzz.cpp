#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <cstring>
#include <vector>
#include <iostream>
#include <array>
#include <algorithm>

namespace tf_fuzzer_utils {
void logError(const std::string& message, const uint8_t* /*data*/, size_t /*size*/) {
    std::cerr << "Error: " << message << std::endl;
}
}  // namespace tf_fuzzer_utils

std::string ConsumeString(const uint8_t* data, size_t& offset, size_t size) {
    if (offset >= size) return "file";
    const size_t remaining = size - offset;
    size_t len = std::min<size_t>((data[offset] % 16) + 1, remaining);
    offset++;
    len = std::min(len, size - offset);
    std::string out(reinterpret_cast<const char*>(data + offset), len);
    offset += len;
    return out.empty() ? "file" : out;
}

int64_t ConsumeBoundedInt(const uint8_t* data, size_t& offset, size_t size,
                          int64_t min_value, int64_t fallback) {
    int64_t value = fallback;
    if (offset + sizeof(int64_t) <= size) {
        std::memcpy(&value, data + offset, sizeof(int64_t));
        offset += sizeof(int64_t);
    }
    if (value < min_value) {
        value = min_value;
    }
    if (value > (1LL << 20)) {
        value = (1LL << 20);
    }
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size == 0) return 0;

    size_t offset = 0;

    tensorflow::Scope root = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

    try {
        int filename_count = 1;
        if (offset < size) {
            filename_count = static_cast<int>(data[offset++] % 3) + 1;
        }
        tensorflow::Tensor filenames_tensor(tensorflow::DT_STRING,
                                            tensorflow::TensorShape({filename_count}));
        auto filenames_flat = filenames_tensor.flat<tensorflow::tstring>();
        for (int i = 0; i < filename_count; ++i) {
            filenames_flat(i) = ConsumeString(data, offset, size);
        }

        tensorflow::Tensor header_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        header_bytes_tensor.scalar<int64_t>()() =
            ConsumeBoundedInt(data, offset, size, /*min_value=*/0, /*fallback=*/0);

        tensorflow::Tensor record_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        record_bytes_tensor.scalar<int64_t>()() =
            ConsumeBoundedInt(data, offset, size, /*min_value=*/1, /*fallback=*/1);

        tensorflow::Tensor footer_bytes_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        footer_bytes_tensor.scalar<int64_t>()() =
            ConsumeBoundedInt(data, offset, size, /*min_value=*/0, /*fallback=*/0);

        tensorflow::Tensor buffer_size_tensor(tensorflow::DT_INT64, tensorflow::TensorShape({}));
        buffer_size_tensor.scalar<int64_t>()() =
            ConsumeBoundedInt(data, offset, size, /*min_value=*/1, /*fallback=*/256 << 10);

        tensorflow::Tensor compression_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
        static constexpr std::array<const char*, 3> kCompressionTypes = {"", "ZLIB", "GZIP"};
        const size_t compression_index = offset < size ? (data[offset++] % kCompressionTypes.size()) : 0;
        compression_tensor.scalar<tensorflow::tstring>()() = kCompressionTypes[compression_index];

        auto filenames_input = tensorflow::ops::Const(root.WithOpName("filenames"), filenames_tensor);
        auto header_bytes_input = tensorflow::ops::Const(root.WithOpName("header_bytes"), header_bytes_tensor);
        auto record_bytes_input = tensorflow::ops::Const(root.WithOpName("record_bytes"), record_bytes_tensor);
        auto footer_bytes_input = tensorflow::ops::Const(root.WithOpName("footer_bytes"), footer_bytes_tensor);
        auto buffer_size_input = tensorflow::ops::Const(root.WithOpName("buffer_size"), buffer_size_tensor);
        auto compression_input = tensorflow::ops::Const(root.WithOpName("compression_type"), compression_tensor);

        tensorflow::Node* dataset_node = nullptr;
        auto status = tensorflow::NodeBuilder(
                          root.GetUniqueNameForOp("FixedLengthRecordDatasetV2"), "FixedLengthRecordDatasetV2")
                          .Input(filenames_input.node())
                          .Input(header_bytes_input.node())
                          .Input(record_bytes_input.node())
                          .Input(footer_bytes_input.node())
                          .Input(buffer_size_input.node())
                          .Input(compression_input.node())
                          .Attr("metadata", "")
                          .Finalize(root.graph(), &dataset_node);
        if (!status.ok()) {
            tf_fuzzer_utils::logError(status.ToString(), data, size);
            return 0;
        }

        tensorflow::ClientSession session(root);
        std::vector<tensorflow::Tensor> outputs;
        status = session.Run({tensorflow::Output(dataset_node, 0)}, &outputs);
        if (!status.ok()) {
            return 0;
        }

    } catch (const std::exception& e) {
        tf_fuzzer_utils::logError("CPU Execution error: " + std::string(e.what()), data, size);
        return -1;
    }

    return 0;
}
