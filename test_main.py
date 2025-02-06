import unittest
from main import (
    parse_cot_and_answer,
    split_cot_into_steps,
    parse_line_ratings,
    update_ratings_in_steps
)


class TestMainUtilities(unittest.TestCase):
    def test_parse_cot_and_answer(self):
        text_with_final = "Reasoning steps...\nFinal Answer: This is the final answer."
        cot, ans = parse_cot_and_answer(text_with_final)
        self.assertIn("Reasoning steps", cot)
        self.assertEqual(ans, "This is the final answer.")

        text_without_final = "Reasoning steps only."
        cot2, ans2 = parse_cot_and_answer(text_without_final)
        self.assertEqual(cot2, ans2, "If 'Final Answer:' is not present, both CoT and ans should be identical.")

    def test_split_cot_into_steps(self):
        cot_text = "Step 1\nStep 2\nStep 3"
        split_text = split_cot_into_steps(cot_text, "0")
        lines = split_text.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertTrue(all("[Rating: 0]" in line for line in lines))

    def test_parse_line_ratings(self):
        steps = (
            "Step 1 reasoning [Rating: 1]\n"
            "Step 2 reasoning [Rating: -1]\n"
            "Step 3 reasoning [Rating: 0]"
        )
        ratings = parse_line_ratings(steps)
        self.assertEqual(ratings, "1, -1, 0")

    def test_update_ratings_in_steps(self):
        steps = (
            "Step 1 reasoning [Rating: 1]\n"
            "Step 2 reasoning [Rating: 0]\n"
            "Step 3 reasoning [Rating: -1]"
        )
        new_ratings = "0, 0, 1"
        updated = update_ratings_in_steps(steps, new_ratings)
        self.assertIn("[Rating: 0]", updated.split("\n")[0])
        self.assertIn("[Rating: 0]", updated.split("\n")[1])
        self.assertIn("[Rating: 1]", updated.split("\n")[2])


if __name__ == "__main__":
    unittest.main()
