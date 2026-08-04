# Chapter 1: The Python Data Model

We can think of the data model as a description of Python as a framework. It formalizes the interfaces of the building blocks of the language itself, such as sequences, functions, iterators, coroutines, context managers, and so on.

## A Pythonic Card Deck
 Example 1.1. A deck as a sequence of playing cards. It demonstrates the power of implementing just two special methods, `__getitem__` and `__len__`.

 ```python
 import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]

    def __len__(self):
        return len(self._cards)
    def __getitem__(self, position):
        return self._cards[position]
 ```
