#ifndef WALK_ITERATOR_CUH
#define WALK_ITERATOR_CUH

#include "step.cuh"

class WalkIterator {
private:
    const int* nodes_;
    const int64_t* timestamps_;
    size_t index_;
    size_t length_;

public:
    // Constructor
    HOST WalkIterator(const int* nodes, const int64_t* timestamps, const size_t index, const size_t length)
        : nodes_(nodes), timestamps_(timestamps), index_(index), length_(length) {}

    // Iterator operations
    Step operator*() const {
        return {nodes_[index_], timestamps_[index_]};
    }

    WalkIterator& operator++() {
        ++index_;
        return *this;
    }

    WalkIterator operator++(int) {
        const WalkIterator temp = *this;
        ++index_;
        return temp;
    }

    bool operator==(const WalkIterator& other) const {
        return index_ == other.index_;
    }

    bool operator!=(const WalkIterator& other) const {
        return index_ != other.index_;
    }
};

#endif // WALK_ITERATOR_CUH
