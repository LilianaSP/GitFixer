#!/usr/bin/env python3
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import json
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Server configuration
SERVER_URL = "http://localhost:7774/chat"

# Test cases from the training dataset
TEST_CASES = [
    {
        "name": "Buffer Processing",
        "input": """
#include <iostream>
#include <cstring>

void processInput(const char* input) {
    const size_t bufferSize = 10;
    char buffer[bufferSize + 1]; // +1 for null terminator
    strncpy(buffer, input, bufferSize);
    buffer[bufferSize] = '\\0'; // Ensure null termination
    std::cout << "Processed input: " << buffer << std::endl;
}

int main() {
    const size_t largeInputSize = 20;
    char largeInput[largeInputSize] = "This is a large input";
    processInput(largeInput);
    return 0;
}
        """.strip(),
        "expected_output": """
#include <iostream>
#include <cstring>

void processInput(const char* input) {
    const size_t bufferSize = 10;
    char buffer[bufferSize + 1]; // +1 for null terminator
    strncpy(buffer, input, bufferSize);
    buffer[bufferSize] = '\\0'; // Ensure null termination
    std::cout << "Processed input: " << buffer << std::endl;
}

int main() {
    const size_t largeInputSize = 20;
    char largeInput[largeInputSize] = "This is a large input";
    processInput(largeInput);
    return 0;
}
        """.strip()
    },
    {
        "name": "Character Array",
        "input": """
#include <iostream>
using namespace std;

int main() {
    const int BUFFER_SIZE = 10;
    char buffer[BUFFER_SIZE + 1]; // +1 for null terminator
    for(int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i] = 'A';
    }
    buffer[BUFFER_SIZE] = '\\0'; // Adding null terminator at the end
    cout << "Buffer content: " << buffer << endl;
    return 0;
}
        """.strip(),
        "expected_output": """
#include <iostream>
using namespace std;

int main() {
    const int BUFFER_SIZE = 10;
    char buffer[BUFFER_SIZE + 1]; // +1 for null terminator
    for(int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i] = 'A';
    }
    buffer[BUFFER_SIZE] = '\\0'; // Adding null terminator at the end
    cout << "Buffer content: " << buffer << endl;
    return 0;
}
        """.strip()
    },
    {
        "name": "String Copy Function",
        "input": """
#include <string.h>

void copyString(char* dest, const char* src, size_t destSize) {
    strncpy(dest, src, destSize);  // Use strncpy instead of strcpy to avoid buffer overflow
    dest[destSize - 1] = '\\0';  // Ensure null termination
}

int main() {
    char largeSource[1000];
    char smallDestination[50];
    memset(largeSource, 'A', sizeof(largeSource));  // Fill with 'A's
    copyString(smallDestination, largeSource, sizeof(smallDestination));  // Pass the size of the destination as an argument
    return 0;
}
        """.strip(),
        "expected_output": """
#include <string.h>

void copyString(char* dest, const char* src, size_t destSize) {
    strncpy(dest, src, destSize);  // Use strncpy instead of strcpy to avoid buffer overflow
    dest[destSize - 1] = '\\0';  // Ensure null termination
}

int main() {
    char largeSource[1000];
    char smallDestination[50];
    memset(largeSource, 'A', sizeof(largeSource));  // Fill with 'A's
    copyString(smallDestination, largeSource, sizeof(smallDestination));  // Pass the size of the destination as an argument
    return 0;
}
        """.strip()
    }
]

def compute_pass_at_k(results: List[List[bool]], k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
    """
    Calculate pass@k metric for different k values.

    Args:
        results: List of boolean lists, where each inner list represents
                the results of multiple attempts for a problem.
        k_values: List of k values for which to calculate pass@k.

    Returns:
        Dictionary with pass@k values for each k.
    """
    pass_at_k = {}
    total_problems = len(results)

    if total_problems == 0:
        logger.warning("No problems to evaluate")
        return {k: 0.0 for k in k_values}

    for k in k_values:
        problems_passed = 0
        for problem_results in results:
            n = len(problem_results)

            if n == 0:
                logger.warning("Problem with no recorded attempts")
                continue

            # Edge case: If we have fewer attempts than k
            if n < k:
                # Consider it a pass if any attempt was successful
                if any(problem_results):
                    problems_passed += 1
            else:
                # For k attempts, we need at least one successful attempt
                if any(problem_results):
                    problems_passed += 1

        # Calculate pass@k as the proportion of problems that passed
        pass_at_k[k] = problems_passed / total_problems
        logger.info(f"Pass@{k}: {pass_at_k[k]:.2f}")

    return pass_at_k

def test_model_on_training_case(test_case: Dict[str, str], num_attempts: int = 10) -> List[bool]:
    """
    Test the model on a specific training case multiple times.
    """
    results = []

    # Create test directory if it doesn't exist
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)

    # Create a temporary file for the test case
    temp_file = test_dir / f"{test_case['name'].replace(' ', '_')}.cpp"

    for attempt in range(num_attempts):
        try:
            logger.info(f"Testing {test_case['name']} - Attempt {attempt + 1}/{num_attempts}")

            # Write the test case to file
            try:
                with open(temp_file, 'w') as f:
                    f.write(test_case['input'])
            except Exception as e:
                logger.error(f"Error writing test file: {e}")
                return [False] * num_attempts

            # Prepare the request data for the chat endpoint
            payload = {
                "prompt": test_case['input'].strip(),
                "system": "Fix the following C++ code:"
            }

            logger.debug(f"Sending request to server with data: {payload}")

            # Send request to the server
            response = requests.post(
                SERVER_URL,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )

            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.debug(f"Full server response: {json.dumps(result, indent=2)}")

                    if "error" in result:
                        logger.error(f"Server returned error: {result['error']}")
                        results.append(False)
                        continue

                    if isinstance(result, dict):
                        predicted_output = result.get("response", "")
                        if predicted_output:
                            success = predicted_output.strip() == test_case['expected_output'].strip()
                            results.append(success)
                            logger.info(f"Attempt {attempt + 1}: {'Successful' if success else 'Failed'}")

                            if not success:
                                logger.debug(f"Expected output:\n{test_case['expected_output']}")
                                logger.debug(f"Predicted output:\n{predicted_output}")
                        else:
                            logger.warning("No response found in server response")
                            results.append(False)
                    else:
                        logger.warning(f"Response is not a dictionary: {type(result)}")
                        results.append(False)
                except json.JSONDecodeError:
                    logger.error("Failed to parse response as JSON")
                    logger.error(f"Raw response content: {response.text}")
                    results.append(False)
            else:
                logger.error(f"Server error: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                results.append(False)

        except Exception as e:
            logger.error(f"Error in attempt {attempt + 1}: {e}")
            results.append(False)

        time.sleep(1)  # Add a small delay between attempts

    # Clean up the temporary file
    try:
        if temp_file.exists():
            temp_file.unlink()
    except Exception as e:
        logger.error(f"Error removing temporary file: {e}")

    return results

def visualize_pass_at_k(pass_at_k: Dict[int, float], title: str = "Training Pass@k Evaluation Results"):
    """
    Visualize pass@k results using matplotlib.

    Args:
        pass_at_k: Dictionary with pass@k values.
        title: Graph title.
    """
    plt.figure(figsize=(10, 6))
    k_values = list(pass_at_k.keys())
    pass_values = list(pass_at_k.values())

    # Create bar chart
    bars = plt.bar(k_values, pass_values, color='skyblue')

    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.xlabel('k (number of attempts)')
    plt.ylabel('Pass@k Score')
    plt.title(title)
    plt.xticks(k_values)
    plt.ylim(0, 1.1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Save the graph
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "training_pass_at_k_results.png")
    plt.close()

def evaluate_training_performance(test_cases: List[Dict[str, str]] = None, num_attempts: int = 10):
    """
    Evaluate model performance on training cases.

    Args:
        test_cases: List of test cases to evaluate
        num_attempts: Number of attempts per problem
    """
    if test_cases is None:
        test_cases = TEST_CASES

    if not test_cases:
        logger.error("No test cases available")
        return

    all_results = []
    total_problems = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTesting problem {i}/{total_problems}: {test_case['name']}")
        results = test_model_on_training_case(test_case, num_attempts)
        all_results.append(results)
        logger.info(f"Results: {results}")

    # Calculate pass@k
    pass_at_k = compute_pass_at_k(all_results)

    # Print results
    logger.info("\nEvaluation Results:")
    for k, score in pass_at_k.items():
        logger.info(f"Pass@{k}: {score:.2f}")

    # Visualize results
    visualize_pass_at_k(pass_at_k, "Model Training Evaluation")

def main():
    evaluate_training_performance()

if __name__ == "__main__":
    main()