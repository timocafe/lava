//
//  decoder.cpp
//  experiment
//
//  Created by timothee.ewart on 5/6/20.
//

#include "ilc/include/decoder.h"
#include "ilc/utils/utils.h"

#include <filesystem>

namespace ilc {

decoder_helper::decoder_helper(const std::string &model_path) {
    control_tf_log();
    model_path_ = std::filesystem::path(model_path);
    const auto models = model_path_builder(model_path_);

    std::for_each(models.begin(), models.end(), [&](const auto &it) {
        const std::string model_path = it.first.native() + "/";
        const std::string model_name = it.second;
        decoder_.add_model(model_path, model_name);
    });
}

void decoder_helper::decode(const std::string &input, const std::string &output) {
    prepare_files(decoder_, input, output); // prepare input
    decoder_();                             // execute the TBB decoder
}

void decoder_helper::ntokens(const std::size_t tokens) { decoder_.ntokens() = tokens; }

void decoder_helper::gpu(const std::size_t gpu) { decoder_.add_gpu_devices(gpu); }

void decoder_helper::set_png_compression_lvl(const int l) { decoder_.lvl_compression() = l; }

const int decoder_helper::get_png_compression_lvl() const { return decoder_.lvl_compression(); }

} // namespace ilc
