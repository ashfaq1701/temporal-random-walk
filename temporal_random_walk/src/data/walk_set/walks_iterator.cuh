#ifndef WALKS_ITERATOR_CUH
#define WALKS_ITERATOR_CUH

#include "walk.cuh"

// Forward declaration
struct WalkSet;

// Iterator for all non-empty walks in the WalkSet
class WalksIterator {
private:
    const WalkSet* walk_set_;
    size_t walk_index_;
    size_t max_walks_;

    // Helper function
    HOST void find_next_non_empty();

public:
    // Constructor
    explicit WalksIterator(const WalkSet* walk_set, size_t start_index = 0);

    // Iterator operations
    Walk operator*() const;

    WalksIterator& operator++();

    WalksIterator operator++(int);

    bool operator==(const WalksIterator& other) const;

    HOST bool operator!=(const WalksIterator& other) const;
};

#endif // WALKS_ITERATOR_CUH
