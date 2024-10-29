class MemoryBank:
    def __init__(self, size):
        self.size = size
        self.memory = []

    def add(self, item):
        if len(self.memory) >= self.size:
            self.memory.pop(0)
        self.memory.append(item)

    def get(self, index):
        if index < 0 or index >= len(self.memory):
            raise IndexError("Index out of range")
        return self.memory[index]

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return f"MemoryBank(size={self.size}, memory={self.memory})"


    def clear(self):
        self.memory = []

    def is_full(self):
        return len(self.memory) >= self.size

    def get_latest(self):
        if not self.memory:
            raise IndexError("Memory is empty")
        return self.memory[-1]

    def get_oldest(self):
        if not self.memory:
            raise IndexError("Memory is empty")
        return self.memory[0]

    def find(self, condition):
        return [item for item in self.memory if condition(item)]

    def remove(self, item):
        if item in self.memory:
            self.memory.remove(item)
        else:
            raise ValueError("Item not found in memory")

    def __contains__(self, item):
        return item in self.memory

    def __iter__(self):
        return iter(self.memory)
