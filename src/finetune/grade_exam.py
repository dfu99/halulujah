from openai import OpenAI
from dotenv import load_dotenv
import os, json
    
if __name__ == "__main__":
    # Retrieve the API key from env
    load_dotenv(dotenv_path=".env")
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=openai_api_key)

    exam_files = os.listdir("archive/EGNIVIA_EXAM_20250919/")
    # Grade each exam
    for exam_file in exam_files:
        results = json.load(open(os.path.join("archive/EGNIVIA_EXAM_20250919/", exam_file), 'r'))
        print(f"Loaded exam answers from {exam_file}.")
        graded_results = []
        for entry in results:
            question = entry['question']
            expected_answer = entry['expected_answer']
            model_answer = entry['model_answer']

            grading_prompt = """You are a strict grader.
    Given the question, the expected answer, and the model's answer, decide if the model's answer matches the expected answer.

    Answer format: Correct | Incorrect
    Respond with exactly one of these words and nothing else.

    Example:
    Question: Who wrote Hamlet?
    Expected Answer: Shakespeare
    Model's Answer: Shakespeare
    Grader's Response: Correct\n\n"""
            grade_this=f"""Now grade this case:
    Question: {question}\n
    Expected Answer: {expected_answer}\n
    Model's Answer: {model_answer}\n
    Grader's Response:"""

            prompt = grading_prompt + grade_this
            response = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {"role": "system", "content": grading_prompt},
                {"role": "user", "content": grade_this}
                ]
            )
            print(grade_this)
            print(response.choices[0].message.content)

            # Extract 'Correct' or 'Incorrect' from the response
            if "Correct" in response.choices[0].message.content:
                score = "Correct"
            elif "Incorrect" in response.choices[0].message.content:
                score = "Incorrect"
            else:
                score = "Unclear"

            entry['score'] = score

            graded_results.append(entry)
            os.makedirs("exam_results", exist_ok=True)
            json.dump(graded_results, open("exam_results/"+os.path.basename(exam_file)+"_graded.json", 'w'), indent=4)