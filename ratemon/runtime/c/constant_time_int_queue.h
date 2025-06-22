#pragma once
// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

#include <list>
#include <stdexcept>
#include <unordered_map>

// A queue that allows constant time search and deletion of elements. Uses both
// a list and a map to achieve this. The list maintains the order of elements,
// while the map allows for O(1) access to elements for deletion.
class ConstantTimeIntQueue {
private:
  std::list<int> list;
  std::unordered_map<int, std::list<int>::iterator> map;

public:
  void enqueue(const int &val) {
    list.push_back(val);
    map[val] = --list.end(); // Store iterator to newly added element
  }

  int dequeue() {
    if (list.empty()) {
      throw std::runtime_error("Queue is empty.");
    }
    int front = list.front();
    find_and_delete(front); // Remove from map
    return front;
  }

  bool find_and_delete(int val) {
    auto it = map.find(val);
    if (it != map.end()) {
      list.erase(
          it->second); // Erase from list using stored iterator (constant time)
      map.erase(it);   // Erase from map (average constant time)
      return true;
    }
    return false;
  }

  bool contains(int val) { return map.find(val) != map.end(); }

  bool empty() const { return list.empty(); }

  size_t size() const { return list.size(); }
};