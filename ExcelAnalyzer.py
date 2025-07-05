import pandas as pd
import sqlite3
import json
import re
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class MetadataResponse(BaseModel):
    """Model for metadata response"""
    table_name: str = Field(description="Suggested table name in snake_case")
    description: str = Field(description="Description of the table content")
    suggested_primary_key: str = Field(default=None, description="Suggested primary key column name")
    category: str = Field(default="unknown", description="Category of the table (e.g., financial, sales, inventory, etc.)")

class DataTypeResponse(BaseModel):
    """Model for data type response"""
    sql_type: str = Field(description="SQL data type (TEXT, INTEGER, REAL, DATE, BOOLEAN)")
    python_type: str = Field(description="Python data type (str, int, float, datetime, bool)")
    description: str = Field(description="Description of the content")
    constraints: List[str] = Field(default_factory=list, description="List of constraints (e.g., UNIQUE, NOT NULL)")
    is_nullable: bool = Field(default=True, description="Whether the column can be NULL")
    suggested_index: bool = Field(default=False, description="Whether an index is suggested for this column")

class ExcelAnalyzer:
    """Main class for analyzing Excel files and converting them into SQL databases"""
    
    def __init__(self, llm_model="gpt-4.1-mini", db_path="excel_data.db"):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.metadata_parser = PydanticOutputParser(pydantic_object=MetadataResponse)
        self.datatype_parser = PydanticOutputParser(pydantic_object=DataTypeResponse)

        self.setup_prompts()
    
    def setup_prompts(self):
        """Sets up the prompts for LLM analysis"""

        # Prompt for metadata analysis
        self.metadata_prompt = PromptTemplate(
            input_variables=["sheet_name", "columns"],
            partial_variables={
                "format_instructions": self.metadata_parser.get_format_instructions()
            },
            template="""
            Analyze the following metadata of an Excel sheet:
            
            Sheet name: {sheet_name}
            Columns: {columns}
            
            Based on the column names, please return the following information:
             - suggested_table_name
             - description of the table content
             - primary_key_column_name_if_present
             - category (e.g., financial, sales, inventory, etc.)

            {format_instructions}
            
            Use table names in snake_case and in English.
            """
        )
        
        # Prompt for data type analysis
        self.datatype_prompt = PromptTemplate(
            input_variables=["column_name", "sample_data", "unique_values"],
            partial_variables={
                "format_instructions": self.datatype_parser.get_format_instructions()
            },
            template="""
            Analyze the following data column:
            
            Column name: {column_name}
            Sample data: {sample_data}
            Unique values (first 10): {unique_values}
            
            Determine the most appropriate SQL data type and respond with the following information:
             - sql_type: (TEXT, INTEGER, REAL, DATE, BOOLEAN)
             - python_type: # e.g., str, int, float, datetime, bool
             - description of the content: # e.g., "This column contains integer values representing sales figures"
             - list of constraints: # e.g., UNIQUE, NOT NULL
             - is_nullable: a flag indicating whether the column can be NULL
             - suggested_index: a flag indicating whether an index is suggested for this column

            {format_instructions}
            
            Consider:
            - INTEGER for integer numbers
            - REAL for decimal numbers
            - TEXT for strings
            - DATE for dates (format YYYY-MM-DD)
            - BOOLEAN for boolean values
            """
        )
        
        # Prompt for SQL query generation
        self.sql_prompt = PromptTemplate(
            input_variables=["question", "schema", "sample_data"],
            template="""
            Generate an SQL query to answer the following question:
            
            Question: {question}
            
            Database schema:
            {schema}
            
            Sample data:
            {sample_data}
            
            Generate ONLY the SQL query without any additional explanations.
            Use standard SQLite syntax.
            """
        )
    
    def analyze_excel_file(self, file_path: str) -> Dict[str, Any]:
        """Analyzes an Excel file and returns its metadata"""
        logger.info(f"Analyzying: {file_path}")
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets_info = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Analyze sheet metadata
                metadata: MetadataResponse = self._analyze_sheet_metadata(sheet_name, df.columns.tolist())
                
                # Analyze data types for each column
                column_types = {}
                for col in df.columns:
                    col_type: DataTypeResponse = self._analyze_column_type(col, df[col])
                    column_types[col] = col_type
                
                sheets_info[sheet_name] = {
                    'metadata': metadata,
                    'column_types': column_types,
                    'dataframe': df,
                    'shape': df.shape
                }
            
            return sheets_info
            
        except Exception as e:
            logger.error(f"Error while analyzing Excel file {file_path}: {str(e)}")
            raise
    
    def _analyze_sheet_metadata(self, sheet_name: str, columns: List[str]) -> Dict[str, Any]:
        """Analyzes the metadata of a sheet using LLM"""
        chain = self.metadata_prompt | self.llm | self.metadata_parser

        input = {
            "sheet_name": sheet_name,
            "columns": ", ".join(columns)
        }
        
        try:
            result = chain.invoke(input=input)
            assert isinstance(result, MetadataResponse), "Result is not of type MetadataResponse"
            return result.model_dump()
            
        except Exception as e:
            logger.warning(f"Error analyzing metadata for {sheet_name}: {str(e)}")
            return {
                "table_name": sheet_name.lower().replace(' ', '_'),
                "description": f"Table generated from sheet {sheet_name}",
                "suggested_primary_key": None,
                "category": "unknown"
            }
    
    def _analyze_column_type(self, column_name: str, column_data: pd.Series) -> Dict[str, Any]:
        """Analyze the data type of a column using LLM"""
        
        # Prepare sample data
        sample_data = column_data.dropna().head(100).tolist()
        unique_values = column_data.dropna().unique()[:100].tolist()
        
        chain = self.datatype_prompt | self.llm | self.datatype_parser

        input = {
            "column_name": column_name,
            "sample_data": str(sample_data),
            "unique_values": str(unique_values)
        }
        
        try:
            result = chain.invoke(input=input)
            
            assert isinstance(result, DataTypeResponse), "Result is not of type DataTypeResponse"
            
            return result.model_dump()
            
            
        except Exception as e:
            logger.warning(f"Error analyzing type for {column_name}: {str(e)}")
            # Fallback with automatic analysis
            return self._auto_detect_type(column_data)
    
    def _auto_detect_type(self, column_data: pd.Series) -> Dict[str, Any]:
        """Fallback for automatic column type detection"""
        if pd.api.types.is_integer_dtype(column_data):
            return {"sql_type": "INTEGER", "python_type": "int", "is_nullable": column_data.isna().any()}
        elif pd.api.types.is_float_dtype(column_data):
            return {"sql_type": "REAL", "python_type": "float", "is_nullable": column_data.isna().any()}
        elif pd.api.types.is_datetime64_any_dtype(column_data):
            return {"sql_type": "DATE", "python_type": "datetime", "is_nullable": column_data.isna().any()}
        elif pd.api.types.is_bool_dtype(column_data):
            return {"sql_type": "BOOLEAN", "python_type": "bool", "is_nullable": column_data.isna().any()}
        else:
            return {"sql_type": "TEXT", "python_type": "str", "is_nullable": column_data.isna().any()}
    
    def create_tables_and_insert_data(self, sheets_info: Dict[str, Any]) -> Dict[str, str]:
        """Creates the tables in the database and inserts the data"""
        table_names = {}
        
        for sheet_name, info in sheets_info.items():
            table_name = info['metadata']['table_name']
            df = info['dataframe']
            column_types = info['column_types']
            
            # DDL Generation
            ddl = self._generate_create_table_ddl(table_name, df.columns, column_types)
            
            # Table Creation
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.execute(ddl)
            
            # Data insertion
            self._insert_data(table_name, df, column_types)
            
            table_names[sheet_name] = table_name
            logger.info(f"Table {table_name} created and populated with {len(df)} rows")
        
        self.conn.commit()
        return table_names
    
    def _generate_create_table_ddl(self, table_name: str, columns: List[str], 
                                   column_types: Dict[str, Dict]) -> str:
        """Generates the DDL to create a table"""
        ddl_parts = [f"CREATE TABLE {table_name} ("]
        
        for col in columns:
            col_info = column_types[col]
            sql_type = col_info.get('sql_type', 'TEXT')
            nullable = '' if col_info.get('is_nullable', True) else ' NOT NULL'
            
            # Sanitize column name
            clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', col.lower())
            ddl_parts.append(f"    {clean_col} {sql_type}{nullable},")
        
        # Remove trailing comma and close
        ddl_parts[-1] = ddl_parts[-1].rstrip(',')
        ddl_parts.append(")")
        
        return '\n'.join(ddl_parts)
    
    def _insert_data(self, table_name: str, df: pd.DataFrame, column_types: Dict[str, Dict]):
        """Inserts the data into the table"""
        
        # Sanitize column names
        clean_columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col.lower()) for col in df.columns]
        
        # Prepare data for insertion
        df_clean = df.copy()
        df_clean.columns = clean_columns
        
        # Convert data types
        for orig_col, clean_col in zip(df.columns, clean_columns):
            col_type = column_types[orig_col].get('sql_type', 'TEXT')
            
            if col_type == 'DATE':
                df_clean[clean_col] = pd.to_datetime(df_clean[clean_col], errors='coerce')
            elif col_type == 'INTEGER':
                df_clean[clean_col] = pd.to_numeric(df_clean[clean_col], errors='coerce').astype('Int64')
            elif col_type == 'REAL':
                df_clean[clean_col] = pd.to_numeric(df_clean[clean_col], errors='coerce')
            elif col_type == 'BOOLEAN':
                df_clean[clean_col] = df_clean[clean_col].astype('boolean')
        
        df_clean.to_sql(table_name, self.conn, if_exists='append', index=False)
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Get the schema of the SQLite database"""
        cursor = self.conn.cursor()
        
        # Get the table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            # Get the schema of the table
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Get some sample data from the table
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            sample_data = cursor.fetchall()
            
            schema[table] = {
                'columns': columns,
                'sample_data': sample_data
            }
        
        return schema
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """It answers a question using the LLM and the database schema"""
        
        # Get the database schema
        schema = self.get_database_schema()
        
        # Prepare schema and sample data for LLM
        schema_text = self._format_schema_for_llm(schema)
        sample_data_text = self._format_sample_data_for_llm(schema)
        
        # Generate a SQL query using the LLM
        chain = self.sql_prompt | self.llm

        input = {
            "question": question,
            "schema": schema_text,
            "sample_data": sample_data_text
        }
        
        try:
            sql_query:AIMessage = chain.invoke(input=input)

            # print(f"Generated SQL Query: {sql_query.content}")
            
            # Clean up the SQL query
            sql_query = sql_query.content.replace('```sql', '').replace('```', '').strip()
            
            # Execute queries
            cursor = self.conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            return {
                'question': question,
                'sql_query': sql_query,
                'results': results,
                'column_names': column_names,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error executing the query: {str(e)}")
            return {
                'question': question,
                'sql_query': sql_query if 'sql_query' in locals() else None,
                'error': str(e),
                'success': False
            }
    
    def _format_schema_for_llm(self, schema: Dict[str, Any]) -> str:
        """Formats the schema for the LLM"""
        schema_parts = []
        
        for table_name, table_info in schema.items():
            columns_info = []
            for col_info in table_info['columns']:
                col_name, col_type = col_info[1], col_info[2]
                columns_info.append(f"{col_name} ({col_type})")
            
            schema_parts.append(f"Table {table_name}: {', '.join(columns_info)}")
        
        return '\n'.join(schema_parts)
    
    def _format_sample_data_for_llm(self, schema: Dict[str, Any]) -> str:
        """Formats the sample data for the LLM"""
        sample_parts = []
        
        for table_name, table_info in schema.items():
            if table_info['sample_data']:
                sample_parts.append(f"Esempi da {table_name}:")
                for row in table_info['sample_data'][:2]:  # Solo prime 2 righe
                    sample_parts.append(f"  {row}")
        
        return '\n'.join(sample_parts)
    
    def close(self):
        """close the connection to the database"""
        self.conn.close()