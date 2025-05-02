#ifndef WALK_CUH
#define WALK_CUH

#include "step.cuh"
#include "walk_iterator.cuh"

// Class representing a single walk (non-owning view)
class Walk {
private:
    const int* nodes_;
    const int64_t* timestamps_;
    size_t length_;

public:
    // Constructor
    HOST Walk(const int* nodes, const int64_t* timestamps, const size_t length)
        : nodes_(nodes), timestamps_(timestamps), length_(length) {}

    // Access operations
    Step operator[](const size_t index) const {
        if (index >= length_) {
            throw std::out_of_range("Walk index out of range");
        }
        return {nodes_[index], timestamps_[index]};
    }

    size_t size() const {
        return length_;
    }

    bool empty() const {
        return length_ == 0;
    }

    // Iterator support
    WalkIterator begin() const {
        return {nodes_, timestamps_, 0, length_};
    }

    WalkIterator end() const {
        return {nodes_, timestamps_, length_, length_};
    }

    // Direct access
    const int* nodes() const {
        return nodes_;
    }

    const int64_t* timestamps() const {
        return timestamps_;
    }

    HOST Step back() const {
        if (length_ == 0) {
            throw std::out_of_range("Walk is empty");
        }
        return {nodes_[length_ - 1], timestamps_[length_ - 1]};
    }

    HOST Step front() const {
        if (length_ == 0) {
            throw std::out_of_range("Walk is empty");
        }
        return {nodes_[0], timestamps_[0]};
    }
};

#endif // WALK_CUH
