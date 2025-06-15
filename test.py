import pandas as pd
import random

# Assuming term_to_cat dict exists in your notebook scope.
# For demonstration, re-defining a small sample; replace with your full dict.
term_to_cat = {
    "mentor": "mentorship",
    "supervisor": "mentorship",
    "workshop": "learning_opportunities",
    "deadline": "workload",
    "team": "team_collaboration",
    "workspace": "environment",
    "networking": "professional_networking",
    "career": "career_guidance"
}

terms = list(term_to_cat.keys())
records = []
for i in range(10000):
    term = random.choice(terms)
    records.append({
        "id": i,
        "term": term,
        "category": term_to_cat[term],
        "polarity": random.choice([-1, 0, 1])
    })

test_df = pd.DataFrame(records)
test_df.to_csv("absa_test.csv", index=False)

# Display a snippet and confirm shape
test_df.head(), test_df.shape
