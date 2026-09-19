import requests

questions = [
    {
        "question": "How does FastAPI handle data validation?",
        "category": "backend",
        "expected_answer": "Pydantic",
        "expected_source": "engineering",
        "should_know": True
    },
    {
        "question": "What is FastAPI?",
        "category": "backend",
        "expected_answer": "FastAPI",
        "expected_source": "engineering",
        "should_know": True
    },
    {
        "question": "What is revenue?",
        "category": "finance",
        "expected_answer": "income",
        "expected_source": "finance",
        "should_know": True
    },
    {
        "question": "What are the three major categories of cash flow?",
        "category": "finance",
        "expected_answer": "operating",
        "expected_source": "finance",
        "should_know": True
    },
    {
        "question": "Who invented the telephone?",
        "category": "finance",
        "expected_answer": "I don't know",
        "expected_source": None,
        "should_know": False
    },
    {
        "question": "How does the API make sure incoming data is valid?",
        "category": "backend",
        "expected_answer": "Pydantic",
        "expected_source": "engineering",
        "should_know": True
    },
    {
        "question": "What money does a company get from selling its products?",
        "category": "finance",
        "expected_answer": "revenue",
        "expected_source": "finance",
        "should_know": True
    },
    {
        "question": "What are the different ways cash moves in financial reporting?",
        "category": "finance",
        "expected_answer": "operating",
        "expected_source": "finance",
        "should_know": True
    },
    {
    "question": "What database supports foreign keys and complex SQL queries?",
    "category": "backend",
    "expected_answer": "PostgreSQL",
    "expected_source": "engineering",
    "should_know": True
    },
]

total = len(questions)
passed = 0

for test in questions:

    response = requests.post(
        "http://127.0.0.1:8000/query",
        json={
            "question": test["question"],
            "category": test["category"]
        }
    )

    data = response.json()

    answer = data["answer"]
    sources = data["sources"]
    best_score = data["best_score"]

    print(f"Best reranker score: {best_score:.4f}")

    # -------------------------
    # Answer evaluation
    # -------------------------

    answer_pass = (
        test["expected_answer"].lower()
        in answer.lower()
    )

    # -------------------------
    # Source evaluation
    # -------------------------

    source_pass = True

    if test["expected_source"]:

        source_pass = any(
            test["expected_source"].lower()
            in source["metadata"].get(
                "document_id", ""
            ).lower()
            for source in sources
        )

    # -------------------------
    # Final result
    # -------------------------

    test_pass = answer_pass and source_pass

    if test_pass:
        passed += 1

    status = "PASS" if test_pass else "FAIL"

    print("\n------------------------------")
    print(f"Question: {test['question']}")
    print(f"Expected answer: {test['expected_answer']}")
    print(f"Answer: {answer}")
    print(f"Expected source: {test['expected_source']}")
    print(f"Sources returned: {len(sources)}")
    print(f"Answer check: {'PASS' if answer_pass else 'FAIL'}")
    print(f"Source check: {'PASS' if source_pass else 'FAIL'}")
    print(f"Result: {status}")


print("\n==============================")
print("EVALUATION SUMMARY")
print("==============================")
print(f"Passed: {passed}/{total}")
print(f"Accuracy: {(passed / total) * 100:.1f}%")