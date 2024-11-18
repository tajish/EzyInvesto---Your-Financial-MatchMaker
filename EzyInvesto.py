import streamlit as st
import ollama
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load JSON Database
def load_json_database(file_path):
    database = {}
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line.strip())
                key = json_object.get('question')  # Replace 'key' with the relevant field in your JSONL
                if key:
                    database[key] = json_object
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON line: {line}. Error: {e}")
    return database


# Function to query JSON Database
def query_json_database(prompt, database):
    for key, value in database.items():
        if prompt.lower() in key.lower():  # Simple keyword-based search
            return value
    return None

# Load CSV Database
@st.cache_data
def load_csv_database(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

# Function to automatically plot data based on query
def auto_plot_csv_data(df, prompt):
    # Example: If prompt contains "sector", generate a sector-based bar chart
    if "funding" in prompt.lower() and "Firm_Name_Type" in df.columns and "Funding_Start" in df.columns:
        plt.figure(figsize=(10, 5))
        sector_data = df.groupby("Firm_Name_Type")["Funding_Start"].sum()
        sector_data.plot(kind="bar")
        plt.title("Funding Distribution by Firm Names")
        plt.xlabel("Firm Name")
        plt.ylabel("Total Funding")
        st.pyplot(plt)
    elif "stage" in prompt.lower() and "Stage" in df.columns and "Funding" in df.columns:
        plt.figure(figsize=(10, 5))
        stage_data = df.groupby("Stage")["Funding"].sum()
        stage_data.plot(kind="bar")
        plt.title("Funding Distribution by Stage")
        plt.xlabel("Stage")
        plt.ylabel("Total Funding")
        st.pyplot(plt)
    else:
        st.write("No relevant visualization available for this query.")

# Function to provide template-based explanation
def workflow_explanation():
    explanation = """
    You are an AI assistant helping startups find the most relevant investors.
    Use the {prompt} to answer the query. Ensure your answer includes:

    1. A list of the top 3 investors name {database} based on the query's requirements.Be descriptive about why you are selecting those.
    2. A rationale for each investor recommendation and its details.Be Descriptive and broad.
    3. Actionable next steps for the startup.

    Based on the {prompt},show some plottings at the end of your response using the csv file that I gave you

    ### Your Response:
    """
    return explanation

# Load the JSON and CSV files
json_database = load_json_database('Financial_Recommendation.jsonl')
csv_file_path = 'Trimmed_Investor_Details_Dataset.csv'  # Replace with the actual CSV file path
csv_database = load_csv_database(csv_file_path)

st.title("💬 WELCOME TO EzyINVESTO")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

# Function to generate a combined database context
def build_database_context(prompt, json_database, csv_database):
    context = ""

    # Query JSON Database
    json_response = query_json_database(prompt, json_database)
    if json_response:
        context += f"JSON Database Info:\n{json_response}\n"

    # Query CSV Database
    if csv_database is not None:
        if "funding" in prompt.lower() and "Firm_Name_Type" in csv_database.columns and "Funding_Start" in csv_database.columns:
            top_firms = csv_database.sort_values(by="Funding_Start", ascending=False).head(3)
            context += "CSV Database Info: Top 3 Firms by Funding:\n"
            for _, row in top_firms.iterrows():
                context += f"- {row['Firm_Name_Type']}: Funding starts at {row['Funding_Start']}\n"
    
    return context

def generate_response_with_database_and_visualization(prompt, json_database, csv_database):
    # Build the database context
    database_context = build_database_context(prompt, json_database, csv_database)

    # Add database context to the input messages
    input_messages = st.session_state.messages + [
        {"role": "system", "content": workflow_explanation()},
        {"role": "system", "content": f"Database Context:\n{database_context}"},
        {"role": "system", "content": prompt}
    ]

    # Generate the response
    response = ollama.chat(model='qwen2.5:0.5b', stream=True, messages=input_messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token

    # Automatically generate visualization if CSV database is loaded
    if csv_database is not None:
        auto_plot_csv_data(csv_database, prompt)

# Chat Input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    if prompt.strip().lower() == "funding":
        explanation = workflow_explanation()
        st.session_state["full_message"] = explanation
        st.chat_message("assistant", avatar="🤖").write(explanation)
        st.session_state.messages.append({"role": "assistant", "content": explanation})
    else:
        # Generate response with integrated database context and visualization
        st.session_state["full_message"] = ""
        st.chat_message("assistant", avatar="🤖").write_stream(
            generate_response_with_database_and_visualization(prompt, json_database, csv_database)
        )
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})

