# Script to list custom Python requirements
import os

# Path to your requirements.txt file
requirements_file = '/Users/bilalchoudhary/Desktop/ura-w25/etis-dag/airflow_home/requirements.txt'

# Standard Python libraries (you can expand this list as needed)
standard_libraries = [
    'airflow', 'flask', 'requests', 'numpy', 'pandas', 'scipy', 'matplotlib',
    'scikit-learn', 'tensorflow', 'pytest', 'sqlalchemy', 'django', 'flask',
    'boto3', 'pyyaml', 'jinja2', 'cryptography', 'gunicorn'
]

# Read the requirements.txt file
if os.path.exists(requirements_file):
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()

    # Strip whitespace and filter out comments
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

    # Identify custom requirements
    custom_requirements = [req for req in requirements if not any(lib in req for lib in standard_libraries)]

    print("Custom requirements found in the repository:")
    for custom_req in custom_requirements:
        print(custom_req)
else:
    print(f"File not found: {requirements_file}")
