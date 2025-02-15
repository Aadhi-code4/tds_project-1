# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "python-dateutil",
#     "httpx",
#     "fastapi[all]",
#     "numpy",
#     "uvicorn",
#     "Pillow",
#     "speechrecognition",
#     "requests",
#     "markdown",
#     "asyncio",
#     "aiofiles"
# ]
# ///


import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import sqlite3
import requests
import json
import os
import markdown
import subprocess
from PIL import Image
import speech_recognition as sr
from dateutil.parser import parse
import glob
import numpy as np
import re
import asyncio
import traceback
import base64
from pathlib import Path



# Initialize FastAPI app and configure CORS
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
EMBEDDING_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json",
}
BATCH_SIZE = 1000  # API batching to avoid timeouts
functions_tools = [
    {
        "type": "function", # ---> A1
        "function": {
            "name": "script_runner",
            "description": "Install a package and run a script from a URL with provided arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The URL of the script to run."
                    },
                    "args": {
                            "type": "string",
                            "description": "The first arguments to pass to the script (eg: 23f2004462@ds.study.iitm.ac.in"
                    }
                },
                "required": ["script_url", "args"]
            }
        }
    },
    {
        "type": "function", # ---> A2
        "function": {
            "name": "format_file_with_prettier",
            "description": "Format a file using Prettier with a specific prettier version",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be formatted (e.g., ./data/format.md)"
                    },
                    "prettier_version": {
                        "type": "string",
                        "description": "The version of Prettier to use (e.g., '3.4.2')"
                    }
                },
                "required": ["file_path", "prettier_version"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function", # ---> A3
        "function": { 
            "name": "count_weekdays",
            "description": "Count the number of specific weekdays from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_loc": {
                        "type": "string",
                        "description": "The location of the input file containing dates"
                    },
                    "output_file_loc": {
                        "type": "string",
                        "description": "The location where the output count will be written"
                    },
                    "day": {
                        "type": "string",
                        "description": "The day of the week to count (e.g., 'Monday', 'Tuesday', etc.)"
                    }
                },
                "required": ["input_file_loc", "output_file_loc", "day"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function", # ---> A4
        "function": {
            "name": "sort_contacts",
            "description": "Sort contacts by specified fields and write to an output file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_loc": {
                        "type": "string",
                        "description": "The path to the input JSON file containing contacts (e.g., ./data/contacts.json)"
                    },
                    "output_file_loc": {
                        "type": "string",
                        "description": "The path to the output JSON file where sorted contacts will be written (e.g., ./data/sorted_contacts.json)"
                    },
                    "first_sort": {
                        "type": "string",
                        "description": "The first field to sort by (e.g., 'last_name')"
                    },
                    "second_sort": {
                        "type": "string",
                        "description": "The second field to sort by (e.g., 'first_name')"
                    }
                },
                "required": ["input_file_loc", "output_file_loc", "first_sort", "second_sort"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function", # ---> A5
        "function": {
            "name": "write_recent_logs",
            "description": "Write the specified line from the most recent .log files to an output file",
            "parameters": {
                "type": "object",
                "properties": {
                    "no_of_log": {
                        "type": "integer",
                        "description": "The number of most recent log files to read (e.g., 10)"
                    },
                    "which_line": {
                        "type": "integer",
                        "description": "The line number to extract from each log file (e.g., 1 for the first line)"
                    },
                    "input_file_loc": {
                        "type": "string",
                        "description": "The path to the directory containing the log files (e.g., /data/logs)"
                    },
                    "output_file_loc": {
                        "type": "string",
                        "description": "The path to the output file (e.g., /data/logs-recent.txt)"
                    }
                },
                "required": ["no_of_log", "which_line", "input_file_loc", "output_file_loc"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function", #  ---> A6
        "function": {
            "name": "create_index",
            "description": "Create an index file for Markdown files based on specified tag and occurrence",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_loc": {
                        "type": "string",
                        "description": "The path to the directory containing the files (e.g., /data/docs)"
                    },
                    "output_file_loc": {
                        "type": "string",
                        "description": "The path to the output JSON file (e.g., /data/docs/index.json)"
                    }
                },
                "required": ["input_file_loc", "output_file_loc"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",  #  ---> A7
        "function": {
            "name": "extract_email_from_content",
            "description": "Extract the sender's email address from an input file and write just the email address to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_loc": {
                        "type": "string",
                        "description": "The location of the input file."
                    },
                    "output_file_loc": {
                        "type": "string",
                        "description": "The location of the output file."
                    }
                },
                "required": ["input_file_loc", "output_file_loc"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",  # ---> A8
        "function": {
            "name": "extract_credit_card_number",
            "description": "Extract a credit card number from an image and save it to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The path to the image file (e.g., /data/images/card.png)"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "The path to the output file (e.g., /data/output/card_number.txt)"
                    }
                },
                "required": ["image_path", "output_path"],
                "additionalProperties": False
            },"strict": True
        }
    },
    {
        "type": "function", # ---> A9
        "function": {
            "name": "comments_similarity_tool",
            "description": "Compare comments from a file and save the most similar comments to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "comments_file": {
                        "type": "string",
                        "description": "The path to the file containing the comments (e.g., /data/comments.txt)"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "The path to the output file where the most similar comments will be saved (e.g., /data/most_similar_comments.txt)"
                    }
                },
                "required": ["comments_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",  # ---> A10
        "function": {
            "name": "calculate_gold_ticket_sales",
            "description": "Calculate the total sales for 'Gold' ticket types and write the result to a file",
            "parameters": 
            {    
                "type": "object",
                "properties": 
                {
                    "database_file_loc": 
                    {
                        "type": "string",
                        "description": "The location of the SQLite database file (e.g., /data/ticket-sales.db)"
                    },
                    "output_file_loc": 
                    {
                        "type": "string",
                        "description": "The location where the total sales for 'Gold' ticket types will be written (e.g., /data/ticket-sales-gold.txt)"
                    }
                },
                "required": ["database_file_loc", "output_file_loc"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",  # ---> B2
        "function": {
            "name": "prevent_file_deletion",
            "description": "Data is never deleted anywhere on the file system, even if the task description asks for it",
        }
    }
    ,
    {
        "type": "function",# ---> B4
        "function": {
            "name": "clone_and_commit", 
            "description": "Clone a Git repository, create a new file, commit the changes, and push them to the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "The URL of the Git repository to clone"},
                    "commit_message": {"type": "string", "description": "The commit message for the changes"},
                    "file_name": {"type": "string", "description": "The name of the new file to create in the repository"},
                    "file_content": {"type": "string", "description": "The content to write into the new file"}
                },
                "required": ["repo_url", "commit_message", "file_name", "file_content"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function", #  ---> B3
        "function": {
            "name": "fetch_data_from_api",
            "description": "Fetch data from an API and save it to a specified path",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_url": {"type": "string", "description": "The URL of the API to fetch data from"},
                    "save_path": {"type": "string", "description": "The path to save the fetched data (e.g., /data/data.json)"}
                },
                "required": ["api_url", "save_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function", #  ---> B5
        "function": {
            "name": "run_sql_query",
            "description": "Run an SQL query on a specified SQLite database",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_path": {"type": "string", "description": "The path to the SQLite database file"},
                    "query": {"type": "string", "description": "The SQL query to be executed"}
                },
                "required": ["database_path", "query"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function", #  ---> B7
        "function": {
            "name": "resize_image",
            "description": "Resize an image and save it to a specified output path",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "The path to the input image file"},
                    "output_path": {"type": "string", "description": "The path to save the resized image (e.g., /data/resized_image.jpg)"},
                    "width": {"type": "integer", "description": "The width of the resized image in pixels"},
                    "height": {"type": "integer", "description": "The height of the resized image in pixels"}
                },
                "required": ["image_path", "output_path", "width", "height"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",  #  ---> B8
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe audio from an MP3 file to text and save it to a specified output path",
            "parameters": {
                "type": "object",
                "properties": {
                    "inputfile": {"type": "string", "description": "The path or URL of the input MP3 file"},
                    "outputfile": {"type": "string", "description": "The path to save the transcribed text (e.g., /data/transcription.txt)"}
                },
                "required": ["inputfile", "outputfile"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",# ---> B9
        "function": {
            "name": "convert_markdown_to_html", 
            "description": "Convert a Markdown file to HTML and save it to a specified output path",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_loc": {"type": "string", "description": "The path to the input Markdown file"},
                    "output_loc": {"type": "string", "description": "The path to save the converted HTML file"}
                },
                "required": ["input_loc", "output_loc"],
                "additionalProperties": False
            },
            "strict": True
        }
    } 
]
#  Main function for (A1 code)     -----------------------------
async def script_runner(script_url, email_list):
    if isinstance(email_list, str):
        email_list = [email_list]  # Convert single email to a list
    current_dir = os.path.abspath("./data")
    command = ["uv", "run", script_url, *email_list, "--root", current_dir]

    process = await asyncio.create_subprocess_exec(*command)
    await process.communicate()  # Wait for the process to finish

    return f"Command executed successfully in {current_dir}"

#  Main function for (A2 code)     -----------------------------
def format_file_with_prettier(file_path: str, prettier_version: str):
    input_file_path = file_path    
    # Verify if the file exists
    if not os.path.isfile(input_file_path):
        return(f"File not found: {input_file_path}")
        
    npx_path = "npx"  # Use "npx" since it should be in the PATH if Node.js is set up correctly
    try:
        result = subprocess.run([npx_path, f"prettier@{prettier_version}", "--write", input_file_path], check=True, capture_output=True, text=True, shell=True)
        print(f"Successfully formatted the file {input_file_path} using Prettier {prettier_version}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during formatting: {e}")
        print(e.output)
    except FileNotFoundError as e:
        print(f"File or executable not found: {e}")

#  Main function for (A3 code)     -----------------------------
async def count_weekdays(input_file: str, output_file: str, weekday: str):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if weekday not in weekdays:
        return f"Invalid weekday: {weekday}. Choose from {weekdays}."
    weekday_index = weekdays.index(weekday)
    count = 0
    try:
        with open(input_file, "r") as file:
            for line in file:
                date_str = line.strip()
                if not date_str:
                    continue  # Skip empty lines
                try:
                    parsed_date = parse(date_str)
                    if parsed_date.weekday() == weekday_index:
                        count += 1
                except ValueError:
                    print(f"Skipping invalid date format: {date_str}")
    except Exception as e:
        return f"Error reading input file: {e}"
    wrote = False
    try:
        with open(output_file, "w") as file:
            file.write(str(count))
        wrote = True
    except Exception as e:
        return f"Error writing to output file: {e}"
    return f'File has been written in {output_file} ({wrote}) and Number of {weekday}s: {count}'

#  Main function for (A4 code)     -----------------------------
async def sort_contacts(input_file_loc,output_file_loc,first_sort="last", second_sort="first"):
    input_file_loc = input_file_loc
    output_file_loc = output_file_loc
    with open(input_file_loc, 'r') as file:
        contacts = json.load(file)

    # Sort contacts by last_name, then by first_name
    sorted_contacts = sorted(contacts, key=lambda x: (x[first_sort], x[second_sort]))

    # Write the sorted contacts to the output file
    with open(output_file_loc, 'w') as file:
        json.dump(sorted_contacts, file, indent=4)
    return(f"Contacts sorted are written to this file loc {output_file_loc}")

# Main function for (A5 code)     -----------------------------
def write_recent_logs(input_file,output_file,no_of_log=10, which_line=1):
    input_file_loc = input_file
    output_file_loc = output_file
    log_files = glob.glob(os.path.join(input_file_loc, '*.log'))
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    recent_logs = log_files[:no_of_log]
    
    with open(output_file_loc, 'w', encoding='utf-8') as output_file:
        for log_file in recent_logs:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if which_line - 1 < len(lines):
                    output_file.write(lines[which_line - 1] + '\n')
    return f"The {no_of_log} log are written in this loc {output_file_loc}"

# Main function for (A6 code)     -----------------------------
def create_index(input_dir, output_file):
    index = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("# "): 
                            title = line.lstrip("#").strip()
                            relative_path = os.path.relpath(filepath, input_dir).replace("\\", "/")
                            index[relative_path] = title
                            break
                        
    with open(output_file, "w", encoding="utf-8") as index_file:
        json.dump(index, index_file, indent=4)
    
    return f"The file has been successfully created at {output_file}"

# Main function for (A7 code)     -----------------------------
async def extract_email_from_content(email_content: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Extract the sender's email address from the following email message:\n\n" + email_content}]
            },
        )
    response.raise_for_status()
    response_data = response.json()
    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")# Extract email from response
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', content)# Use regex to extract a valid email address
    return match.group(0) if match else ""

# Main function for (A8 code)     -----------------------------
async def extract_credit_card_number(image_path: str, output_path: str):
    try:
        image_path = Path(image_path)  
        output_path = Path(output_path)
        if not image_path.exists():
            return f"Error: Image file '{image_path}' not found."

        # Read and encode the image in Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Analyze the following image and provide me with any numbers you can find."},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]}
                    ],
                    "max_tokens": 50
                }
            )

            response_data = response.json()
            card_number = (
                response_data.get("choices", [{}])[0]
                .get("message", {}).get("content", "")
                .replace(" ", "")
            )
            contents = "".join(response_data["choices"][0]["message"]['content'])
            if not card_number:
                return "Failed to extract credit card number."
            
            match = re.search(r'\b\d{4} \d{4} \d{4} \d{4}\b', contents)
            # Return the matched 16-digit number if found, else return None
            card = match.group() if match else None

            with open(output_path, "w") as output_file:
                output_file.write(card)
            return f"Credit card number extracted and saved to {output_path}"
    
    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()  

# -------- (A9 sub_codes)
async def load_comments(file_path):
    """Load comments from file asynchronously."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
async def get_embeddings(comments):
    """Fetch embeddings in batches for efficiency."""
    embeddings = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(comments), BATCH_SIZE):
            batch = comments[i : i + BATCH_SIZE]
            response = await client.post(EMBEDDING_URL, headers=HEADERS, json={"model": "text-embedding-3-small", "input": batch})
            response.raise_for_status()
            embeddings.extend([item["embedding"] for item in response.json()["data"]])
    return np.array(embeddings)
def find_most_similar_fast(comments, embeddings):
    """Find the most similar comments using vectorized cosine similarity."""
    similarity_matrix = np.dot(embeddings, embeddings.T) / (
        np.linalg.norm(embeddings, axis=1)[:, None] * np.linalg.norm(embeddings, axis=1)
    )
    np.fill_diagonal(similarity_matrix, -1)  # Ignore self-similarity
    i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    return comments[i], comments[j]
# Main function for (A9 codes)   -----------------------------
async def comments_similarity_tool(comments_file, output_file):
    """Find and save the most similar comments asynchronously."""
    comments_file = comments_file
    output_file = output_file
    comments = await load_comments(comments_file)
    if len(comments) < 2:
        return "Not enough comments to compare."

    embeddings = await get_embeddings(comments)
    comment1, comment2 = find_most_similar_fast(comments, embeddings)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{comment1}\n{comment2}\n")

    return f"Most similar comments saved to {output_file}"

#    Main function for (A10 codes)   -----------------------------
def calculate_gold_ticket_sales(database_file_loc: str, output_file_loc: str):

    conn = sqlite3.connect(database_file_loc)
    cursor = conn.cursor()
    query = "SELECT SUM(units * price) as total_sales FROM tickets WHERE type = 'Gold'"
    cursor.execute(query)
    result = cursor.fetchone()[0]

    with open(output_file_loc, 'w') as file:
        file.write(str(result))
    cursor.close()
    conn.close()

    return result

#B3
async def fetch_data_from_api(api_url: str, save_path: str):
    if not await is_within_data_directory(save_path):
        raise HTTPException(status_code=403, detail="Save path must be within /data directory.")
    response = requests.get(api_url)
    data = response.json()
    async with aiofiles.open(save_path, 'w') as file:
        await file.write(json.dumps(data))
    return {"message": "Data fetched and saved successfully"}

#B4
def clone_and_commit(repo_url: str, commit_message: str, file_name: str, file_content: str):
    try:
        # Get the repo name from the URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        subprocess.run(['git', 'clone', repo_url])
        os.chdir(repo_name)
        with open(file_name, 'w') as file:
            file.write(file_content)
        
        subprocess.run(['git', 'add', file_name])
        subprocess.run(['git', 'commit', '-m', commit_message])
        subprocess.run(['git', 'push'])
        
        return {"message": f"Changes committed and pushed to {repo_url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#B5
async def run_sql_query(database_path: str, query: str):
    if not await is_within_data_directory(database_path):
        raise HTTPException(status_code=403, detail="Database path must be within /data directory.")
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return {"results": results}

#B7
async def resize_image(image_path: str, output_path: str, width: int, height: int):
    if not await is_within_data_directory(image_path) or not await is_within_data_directory(output_path):
        raise HTTPException(status_code=403, detail="Image paths must be within /data directory.")
    image = Image.open(image_path)
    image = image.resize((width, height))
    image.save(output_path)
    return {"message": "Image resized successfully"}

#B8
def transcribe_audio(inputfile, outputfile):
    try:
        # Check if the inputfile is a URL
        if inputfile.startswith('http://') or inputfile.startswith('https://'):
            response = requests.get(inputfile)
            with open('temp_audio.mp3', 'wb') as f:
                f.write(response.content)
            inputfile = 'temp_audio.mp3'
        
        # Convert MP3 to WAV using ffmpeg
        wav_file = 'temp_audio.wav'
        subprocess.run(['ffmpeg', '-i', inputfile, wav_file])

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data)

        with open(outputfile, 'w', encoding='utf-8') as file:
            file.write(transcription)

        os.remove(wav_file)
        if 'temp_audio.mp3' in locals():
            os.remove('temp_audio.mp3')

        print(f"Transcription successful! Saved to {outputfile}")

    except Exception as e:
        print(f"An error occurred: {e}")

#B9
async def convert_markdown_to_html(input_loc: str, output_loc: str):
    if not await is_within_data_directory(input_loc) or not await is_within_data_directory(output_loc):
        raise HTTPException(status_code=403, detail="Paths must be within /data directory.")
    async with aiofiles.open(input_loc, 'r') as file:
        markdown_content = await file.read()
    html_content = markdown.markdown(markdown_content)
    async with aiofiles.open(output_loc, 'w') as file:
        await file.write(html_content)
    return {"message": "Markdown converted to HTML successfully"}

# Chat GPT Query
async def query_gpt(task: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": "wheneven you receive a systme directory location, always make it into a relative path, for example adding a . before it would make it relative path, rest is on  you to manage, i just want the relative path"
                    },
                    {"role": "user", "content": task}
                    ],
                "tools": functions_tools,
                "tool_choice": "auto",
            },
        )
    
    response_data = response.json()
    
    if "choices" not in response_data or not response_data["choices"]:
        raise HTTPException(status_code=500, detail="Invalid response from OpenAI API (no choices).")
        
    return response_data

@app.get("/run")
async def task(task: str = Query(..., description="The task to perform")):
    try:
        result = await query_gpt(task)
        fun = result["choices"][0]["message"]["tool_calls"][0]["function"]
        arguments = json.loads(fun["arguments"])
        
        tool_call = result["choices"][0]["message"]["tool_calls"][0] if "tool_calls" in result["choices"][0]["message"] else None
        function_name = tool_call["function"]["name"]
        raw_arguments = tool_call["function"]["arguments"]
        argument = json.loads(raw_arguments)  
        if tool_call:
            
            if fun["name"] == "script_runner":#A1
                script_url = arguments["script_url"]  
                email = arguments["args"]
                ans = await script_runner(script_url=script_url, email_list=email)
                return ans
            
            elif fun["name"] == "format_file_with_prettier":#A2
                file_path = os.path.abspath(arguments["file_path"])
                prettier_version = arguments["prettier_version"]
                format_file_with_prettier(file_path, prettier_version)
                return f"File formatted successfully {file_path}"
            
            elif fun["name"] == "count_weekdays":#A3
                input_file_loc = os.path.abspath(arguments["input_file_loc"])
                output_file_loc = os.path.abspath(arguments["output_file_loc"])
                day = arguments["day"]
                ans = await count_weekdays(input_file_loc, output_file_loc, day)
                return ans
            
            elif fun["name"] == "sort_contacts":#A4
                input_file_loc = os.path.abspath(arguments["input_file_loc"])
                output_file_loc = os.path.abspath(arguments["output_file_loc"])
                first_sort = arguments["first_sort"]
                second_sort = arguments["second_sort"]
                ans = await sort_contacts(input_file_loc, output_file_loc, first_sort, second_sort)
                return {"result": ans}
            
            elif fun["name"] == "write_recent_logs":#A5
                no_of_log = arguments["no_of_log"]
                which_line = arguments["which_line"]
                input_file_loc = os.path.abspath(arguments["input_file_loc"])
                output_file_loc = os.path.abspath(arguments["output_file_loc"])
                ans = write_recent_logs(input_file_loc, output_file_loc,no_of_log, which_line)
                return {"result": ans}
            
            elif fun["name"] == "create_index":#A6
                input_file_loc = os.path.abspath(arguments["input_file_loc"])
                output_file_loc = os.path.abspath(arguments["output_file_loc"])
                ans = create_index(input_dir=input_file_loc, output_file=output_file_loc)
                return {"result": ans}
            
            elif fun["name"] == "extract_email_from_content":#A7
                input_file = os.path.abspath(arguments["input_file_loc"])
                output_file = os.path.abspath(arguments["output_file_loc"])
                
                with open(input_file, "r", encoding="utf-8") as f:
                    email_content = f.read()
                
                sender_email = await extract_email_from_content(email_content)
                
                if sender_email:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(sender_email)
                    return {"result": sender_email,"loc":f"file has been saved in {output_file}"}
                else:
                    return("No valid email address found.")
                
            elif fun["name"] == "extract_credit_card_number":  # A8
                image_file_path = os.path.abspath(arguments["image_path"])
                output_file_path = os.path.abspath(arguments["output_path"])
                result = await extract_credit_card_number(image_file_path, output_file_path)
                return result 
            
            elif fun["name"] == "comments_similarity_tool":#A9
                comments_file = os.path.abspath(arguments["comments_file"])
                output_file = os.path.abspath(arguments["output_file"])
                ans = await comments_similarity_tool(comments_file,output_file)
                return {"result": ans}
            
            elif fun["name"] == "calculate_gold_ticket_sales":    #A10
                database_file_loc = os.path.abspath(arguments["database_file_loc"])
                output_file_loc = os.path.abspath(arguments["output_file_loc"])
                calculate_gold_ticket_sales(database_file_loc, output_file_loc)
                return f"File formatted successfully in {output_file_loc}"
        
            if function_name == "prevent_file_deletion":
                raise HTTPException(status_code=403, detail="File deletion is not allowed.")

            elif function_name == "convert_markdown_to_html":
                input_file_loc = os.path.abspath(argument["input_loc"])
                output_file_loc = os.path.abspath(argument["output_loc"])
                return await convert_markdown_to_html(input_loc=input_file_loc,output_loc=output_file_loc )
            
            elif function_name == "clone_and_commit":
                repo_url = argument["repo_url"]
                commit_message=argument["commit_message"]
                file_name= argument["file_name"]
                file_content= argument["file_content"]
                return clone_and_commit(repo_url=repo_url, commit_message=commit_message,file_name= file_name,file_content=file_content)
            
            elif function_name == "fetch_data_from_api":
                apiurl=argument["api_url"]
                savepath= os.path.abspath(argument["save_path"])
                return await fetch_data_from_api(api_url=apiurl, save_path=savepath)
            
            elif function_name == "run_sql_query":
                database_path= os.path.abspath(argument["database_path"])
                query=argument["query"]
                return await run_sql_query(database_path=database_path, query=query)
            
            elif function_name == "resize_image":
                image_path= os.path.abspath(argument["image_path"])
                output_path= os.path.abspath(argument["output_path"])
                width=argument["width"]
                height=argument["height"]
                return await resize_image(image_path=image_path, output_path=output_path, width=width, height=height)
            
            elif function_name == "transcribe_audio":
                inputfile= os.path.abspath(argument["inputfile"])
                outputfile=  os.path.abspath(argument["outputfile"])
                return transcribe_audio(inputfile=inputfile,outputfile= outputfile)
            
            else:
                return {"message": "No matching function call."}
        else:
            return {"message": result["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def is_within_data_directory(file_path: str) -> bool:
    data_directory = os.path.abspath("data")
    file_path = os.path.abspath(file_path)
    return os.path.commonpath([data_directory]) == os.path.commonpath([data_directory, file_path])

@app.get("/read")
async def read_file(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
