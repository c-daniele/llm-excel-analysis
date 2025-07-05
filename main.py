from ExcelAnalyzer import *

import os

# Analyzer initialization
analyzer = ExcelAnalyzer()

# sample excel file path
file_path = f"{os.getcwd()}/E0006WW1TQ4.xlsx"

try:
    # Step 1: Analyze the Excel file
    sheets_info = analyzer.analyze_excel_file(file_path)
    
    # Step 2: Create tables and insert data
    table_names = analyzer.create_tables_and_insert_data(sheets_info)
    
    # Step 3: Ask questions
    questions = [
        "How many rows are there in total?",
        "Can you show me the weight of the portfolio for each Country and Sector",
        "Show the top 5 Non-European holdings by weight"
    ]
    
    for question in questions:
        result = analyzer.answer_question(question)
        
        if result['success']:
            print(f"\nQuestion: {question}")
            print(f"SQL Query: {result['sql_query']}")
            # print(f"Results: {result['results']}")

            if 'results' in result:
                df_results = pd.DataFrame(result['results'])
                print(df_results.to_markdown(index=False))

        else:
            print(f"\nError for question '{question}': {result['error']}")
except Exception as e:
        print(f"Error: {str(e)}")
    
finally:
    analyzer.close()