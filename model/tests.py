
import unittest

class TestThings(unittest.TestCase):

    def test_one(self):
        self.assertEqual(1 + 1, 2)

    def test_two(self):
        self.assertEqual(2 + 2, 4)

    def test_three(self):
        self.assertEqual(3 + 3, 5)


if __name__ == "__main__":
    unittest.main()
