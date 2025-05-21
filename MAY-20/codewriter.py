import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def codewriter(prompt: str, file_path: str = "MAY-20/generated_function.py") -> str:
    """Generate code using Gemini AI and save it to a file.
    
    Args:
        prompt (str): The prompt describing the code to generate
        file_path (str, optional): Path where the generated code will be saved. Defaults to "MAY-20/generated_function.py"
    
    Returns:
        str: The generated code or an error message if generation fails
    """
    try:
        response = model.generate_content(prompt)
        with open(file_path, 'w') as f:
            f.write(response.text)
        return response.text
    except Exception as e:
        return f"Error generating code: {str(e)}"



'''if __name__ == "__main__":
    # Example prompt for code generation
    prompt = """Write a Python function that:
    1. Takes a list of numbers as input
    2. Returns the sum of all even numbers in the list"""
    
    # Generate code from the prompt
    generated_code = codewriter(prompt)
    print("\nGenerated Code:")
    print(generated_code)
    
    # Save the generated code to a file
    file_path = "MAY-20/generated_function.py"
    save_result = save_code_to_file(generated_code, file_path)
    
    if isinstance(save_result, str):
        print(f"\nError: {save_result}")
    else:
        print(f"\nCode successfully saved to {file_path}")'''
