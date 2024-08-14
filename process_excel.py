import pandas as pd
from chatbot_function import Chatbot, generate_response
from final import chat_tfidf

def process_excel(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Ensure the 'Input' and 'Output' columns exist
    if 'Input' not in df.columns:
        raise ValueError("The Excel file must have an 'Input' column")
    if 'Output' not in df.columns:
        df['Output'] = ''  # Create the Output column if it doesn't exist
    
    # Read the dataset for the chatbot
    df_chatbot = pd.read_excel("dialog_talk_agent.xlsx")
    
    # Initialize the chatbot
    chatbot = Chatbot(df_chatbot)
    
    # Generate responses for each input
    for index, row in df.iterrows():
        if pd.notna(row['Input']) and pd.isna(row['Output']):
            df.at[index, 'Output'] = chat_tfidf(row['Input'])
    
    # Save the results back to the same Excel file
    df.to_excel(file_path, index=False)
    print(f"Processed {len(df)} rows and updated results in {file_path}")

if __name__ == "__main__":
    file_path = "Book1.xlsx"
    process_excel(file_path)