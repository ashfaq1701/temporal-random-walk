#ifndef WALKS_ITERATOR_IMPL_CUH
#define WALKS_ITERATOR_IMPL_CUH

#include "walks_iterator.cuh"

HOST inline void WalksIterator::find_next_non_empty() {
    while (walk_index_ < max_walks_ && walk_set_->walk_lens[walk_index_] == 0) {
        ++walk_index_;
    }
}

inline WalksIterator::WalksIterator(const WalkSet* walk_set, const size_t start_index)
    : walk_set_(walk_set), walk_index_(start_index), max_walks_(walk_set->num_walks) {
    find_next_non_empty();
}

// Iterator operations
inline Walk WalksIterator::operator*() const {
    size_t len = walk_set_->walk_lens[walk_index_];
    const size_t offset = walk_index_ * walk_set_->max_len;
    return {walk_set_->nodes + offset, walk_set_->timestamps + offset, len};
}

inline WalksIterator& WalksIterator::operator++() {
    ++walk_index_;
    find_next_non_empty();
    return *this;
}

inline WalksIterator WalksIterator::operator++(int) {
    const WalksIterator temp = *this;
    ++(*this);
    return temp;
}

inline bool WalksIterator::operator==(const WalksIterator& other) const {
    return walk_index_ == other.walk_index_;
}

HOST inline bool WalksIterator::operator!=(const WalksIterator& other) const {
    return walk_index_ != other.walk_index_;
}

#endif // WALKS_ITERATOR_IMPL_CUH
