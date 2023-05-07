//
//  Created by timothee.ewart on 5/7/23
//

#ifndef message__h
#define message__h

namespace lava
{

    // tiny helper class to communicate between node
    // it must by using move semantique do not forget std::move
    // to return from the node, else copy

    template <class T>
    struct message
    {
        typedef T value_type;

        explicit message(const bool empty = true)
            : empty_(empty), data_(std::vector<value_type>()), meta_(std::map<std::string, uint64_t>()),
              name_(std::string()), model_(std::string()) {}

        void swap_data(std::vector<value_type> &v) { std::swap(data_, v); }

        const bool &empty() const { return empty_; }
        bool &empty() { return empty_; }

        std::vector<value_type> &data() { return data_; }
        const std::vector<value_type> &data() const { return data_; }

        std::map<std::string, uint64_t> &meta() { return meta_; }
        const std::map<std::string, uint64_t> &meta() const { return meta_; }

        const std::string &name() const { return name_; }
        std::string &name() { return name_; }

        const std::string &model() const { return model_; }
        std::string &model() { return model_; }

        bool empty_;
        // all data are saved using std::vector
        std::vector<value_type> data_;
        // some meta data (height, width, etc ...)
        std::map<std::string, uint64_t> meta_;
        // name of the file
        std::string name_;
        // name of the model for onnx
        std::string model_;
    };

} // namespace ilc
#endif // end namespace
