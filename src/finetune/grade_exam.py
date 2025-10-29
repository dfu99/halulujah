from openai import OpenAI
from dotenv import load_dotenv
import os, json
from tqdm import tqdm
    
if __name__ == "__main__":
    # Retrieve the API key from env
    load_dotenv(dotenv_path=".env")
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=openai_api_key)

    # Exam files path
    EXAM_FILES_PATH = "archive/EGNIVIA-ex_EXAM_20251024/"
    # Graded exam files path
    GRADED_EXAM_FILES_PATH = "exam_results/"
    # List all exam files
    exam_files = os.listdir(EXAM_FILES_PATH)
    graded_exam_files = os.listdir(GRADED_EXAM_FILES_PATH) if os.path.exists(GRADED_EXAM_FILES_PATH) else []
    # Filter out already graded files
    exam_files = [f for f in exam_files if os.path.basename(f)+"_graded.json" not in graded_exam_files and not f.endswith(".out")]

    # Print the files to be graded
    print(f"Number of already graded files: {len(graded_exam_files)}")
    print(f"Number of exam files to grade: {len(exam_files)}")

    # Grade each exam
    for exam_file in exam_files:
        results = json.load(open(os.path.join(EXAM_FILES_PATH, exam_file), 'r'))
        print(f"Loaded exam answers from {exam_file}.")
        graded_results = []
        for entry in tqdm(results, desc="Grading entries"):
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
            # print(grade_this)
            # print(response.choices[0].message.content)

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
        print(f"Graded results saved to exam_results/{os.path.basename(exam_file)}_graded.json")