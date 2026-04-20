#ifndef WALKS_ITERATOR_CUH
#define WALKS_ITERATOR_CUH

#include <cstddef>
#include <cstdint>

#include "walk.cuh"

// Iterator over non-empty walks in a host-resident walk buffer set.
// Non-owning: the caller must keep the underlying arrays alive for the
// iterator's lifetime. Decoupled from any particular owning container
// (WalkSetHost today) by taking raw pointers at construction.
class WalksIterator {
private:
    const int*     nodes_;
    const int64_t* timestamps_;
    const size_t*  walk_lens_;
    const int64_t* edge_ids_;
    size_t         max_len_;
    size_t         max_walks_;
    size_t         walk_index_;

    HOST void find_next_non_empty();

public:
    HOST WalksIterator(
        const int*     nodes,
        const int64_t* timestamps,
        const size_t*  walk_lens,
        const int64_t* edge_ids,
        size_t         num_walks,
        size_t         max_len,
        size_t         start_index = 0);

    HOST Walk operator*() const;

    HOST WalksIterator& operator++();
    HOST WalksIterator  operator++(int);

    HOST bool operator==(const WalksIterator& other) const;
    HOST bool operator!=(const WalksIterator& other) const;
};

#endif // WALKS_ITERATOR_CUH
