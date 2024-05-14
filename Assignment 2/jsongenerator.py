# jsongenerator.py

# 2. Importing the built-in JSON package
import json

# 3. Importing from employee.py
from employee import details, employee_name, age, title

def create_dict():
    """Create and return a dictionary with employee information."""
    employee_dict = {
        "first_name": str(employee_name),
        "age": int(age),
        "title": str(title)
    }
    return employee_dict

def write_json_to_file(output_file):
    """Write employee information to a JSON file."""
    # 5. Using json.dumps() to serialize the dictionary to a JSON formatted string
    json_object = json.dumps(create_dict())

    # 6. Writing the JSON string to a file
    with open(output_file, "w") as newfile:
        newfile.write(json_object)

# 9. Running the code
if __name__ == "__main__":
    output_file = "employee_info.json"
    write_json_to_file(output_file)
    print(f"Employee information has been written to {output_file}")
