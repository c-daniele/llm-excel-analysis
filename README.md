# Abstract

As AI developers, we're always looking for ways to make data more accessible and queryable through natural language. While Retrieval-Augmented Generation (RAG) has revolutionized how we interact with unstructired textual documents, **it falls short when dealing with structured data**.
The RAG approach is so powerful that users or even early stage AI developers may fall in **the illusion that it can be applied to any kind of data**, including structured data like Excel files. However, this is a misconception that can lead to frustration and inefficiency.
One of most ubiquitous kind of file asset across all organization is the Excel file format, which could also be considered as structured or "semi-structured" at least.
Anyone who has tryed to process an Excel file using the standard Rag approach, quickly realized there is no real value with processing excel files the same way as PDFs.

I've built a system that combines some prompting techniques to create a powerful Excel analysis tool based on SQL.

# The Solution: LLM-Powered Excel-to-SQL Pipeline

Instead of trying to force RAG into a structured data world, I've built a system that embraces the structured nature of Excel files and uses LLMs to **convert** the excel data **into a SQL database** schema. This allows us to leverage the power of SQL for querying and analyzing the data.

Here's the architecture:

{{< mermaid >}}
graph TD;
    A[Excel File Upload] --> B[LLM Metadata Analysis];
    B --> C[Column Type Detection];
    C --> D[SQL Schema Generation];
    D --> E[Data Insertion];
    E --> F[Ready for Queries];
    
    G[Natural Language Query] --> H[LLM SQL Generation];
    H --> I[Query Execution];
    I --> J[Results & Visualization];
    
    F --> H;
    
    style A fill:#e1f5fe;
    style F fill:#e8f5e8;
    style J fill:#fff3e0;
{{< /mermaid >}}

As for the DB, I used SQLite for simplicity, but this architecture can be adapted to any SQL database.
As for the LLM, I used OpenAI's gpt-4.1-mini, but you can use any comparable LLM.

## System Components

The pipeline consists of four main components:

### 1. **Metadata Analyzer**
Uses an LLM to analyze sheet names and column headers, inferring the purpose and structure of the data:

```python
# Example LLM prompt for metadata analysis
"""
Analyze the following Excel metadata:
Sheet Name: Portfolio_Holdings
Columns: Ticker, Company_Name, Market_Value, Weight_Percent, Sector

Based on column names, suggest:
- Appropriate table name
- Data description
- Primary key candidates
- Data category
"""
```

### 2. **Type Detection Engine**
Combines LLM analysis with statistical sampling to determine the correct SQL data types:

{{< mermaid >}}
graph LR
    A[Sample Data] --> B[LLM Analysis]
    A --> C[Statistical Analysis]
    B --> D[Final Type Decision]
    C --> D
    D --> E[SQL Schema]
{{< /mermaid >}}

At the end of this process, the system generates a SQL schema that accurately represents the data types and relationships in the Excel file and executes it to create the table in the database.

### 3. **SQL Generator**
Converts natural language questions into SQL queries using the database schema as context:

```python
sql_prompt = PromptTemplate(
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
```

### 4. **Query Executor**
Executes the generated SQL and formats results for presentation.

# Real-World Example: ETF Portfolio Analysis

Let me walk you through a concrete example using an XTrackers ETF holdings composition as example file.
The Excel file is pretty simple and contains the breakdown of the "Xtrackers MSCI World ex USA UCITS" ETF, with related underlying stocks, their market value, weight in the portfolio, and sector classification.

## Input Excel File Structure

| ID | Name                  | ISIN          | Country      | Currency | Exchange             | Type of Security | Rating | Industry Classification | Weighting |
|----|-----------------------|---------------|--------------|----------|----------------------|------------------|--------|------------------------|-----------|
| 1  | SAP                   | DE0007164600  | Germany      | EUR      | XETRA                | Equity           | -      | Information Technology | 1.47%     |
| 2  | ASML HOLDING NV       | NL0010273215  | Netherlands  | EUR      | Euronext Amsterdam   | Equity           | Baa2   | Information Technology | 1.46%     |
| 3  | NESTLE SA             | CH0038863350  | Switzerland  | CHF      | Scoach Switzerland   | Equity           | Aa2    | Consumer Staples       | 1.22%     |
| 4  | NOVARTIS AG           | CH0012005267  | Switzerland  | CHF      | Scoach Switzerland   | Equity           | Aa3    | Health Care            | 1.08%     |
| 5  | ROCHE HOLDING PAR AG  | CH0012032048  | Switzerland  | CHF      | Scoach Switzerland   | Equity           | A1     | Health Care            | 1.06%     |
| 6  | ...                   | .....         | ....         | ...      | ...                  | ......           | ..     | ....                   | ...%      |

## System Processing Steps

1. **Metadata Analysis**
   - LLM identifies the dataset as portfolio holdings data
   - Suggests table name: `securities_list`
   - Identifies `ID` as primary key candidate

2. **Type Detection**
   - `ID`: NUMBER (sequence number)
   - `Name`: TEXT (Company Name)
   - `ISIN`: TEXT (Security Identifier)
   - `Country`: TEXT (Country of origin)
   - `Currency`: TEXT (Currency of the security)
   - `Exchange`: TEXT (Trading exchange)
   - `Type of Security`: TEXT (e.g., Equity, Bond)
   - `Rating`: TEXT (Credit rating)
   - `Industry Classification`: TEXT (Sector classification)
   - `Weighting`: REAL (Percentage weight in the portfolio)

3. **SQL Schema Generation**
Automatically generated DDL for the table:

```sql
   CREATE TABLE securities_list (
    id INTEGER NOT NULL,
    name TEXT NOT NULL,
    isin TEXT NOT NULL,
    country TEXT,
    currency TEXT NOT NULL,
    exchange TEXT,
    type_of_security TEXT NOT NULL,
    rating TEXT,
    industry_classification TEXT,
    weighting REAL
);
   ```

1. **Data Insertion**
   Here, the LLM generates automatically the SQL INSERT statements to populate the table with data from the Excel file:
   - Handles format conversion (B for billions, % for percentages)
   - Validates data integrity
   - Inserts all holdings records

## Query Examples

Once processed, users can ask natural language questions:

Let's start with a straightforward question:

---
#### **Query**: "*How many rows are there in total?*"

#### **Generated SQL**
```sql
SELECT COUNT(*) as N FROM securities_list;
```

#### **Output**
|   N |
|----:|
| 796 |

---

Ok, now le'ts see a more complex query that requires aggregation and understanding of the data structure:

#### **Query**: "*Can you show me the weight of the portfolio for each Country and Sector?*"

#### **Generated SQL**
```sql
SELECT country, industry_classification AS sector, SUM(weighting) AS total_weight
FROM securities_list
GROUP BY country, industry_classification;
```

#### **Output**
| country            | industry_classification| total_weight |
|:-------------------|:-----------------------|-------------:|
| -                  | unknown                | 0.00349444   |
| Australia          | Communication Services | 0.00139149   |
| Australia          | Consumer Discretionary | 0.00439915   |
| Australia          | Consumer Staples       | 0.00200737   |
| Australia          | Energy                 | 0.00214571   |
| Australia          | Financials             | 0.0250382    |
| Australia          | Health Care            | 0.00195356   |
| Australia          | Industrials            | 0.00295084   |
| Australia          | Information Technology | 0.000675708  |
| ....               | ....                   | ....         |

---

Now, let's take it a step further and apply some where conditions:

#### **Query**: "*Show me the top 5 Non-European holdings by weight*"

#### **Generated SQL**
```sql
SELECT name, country, weighting
FROM securities_list
WHERE country NOT IN ('Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom')
ORDER BY weighting DESC
LIMIT 5;
```

#### **Output**
| name                               | Country   | weighting  |
|:-----------------------------------|:----------|-----------:|
| COMMONWEALTH BANK OF AUSTRALIA     | Australia | 0.00917773 |
| ROYAL BANK OF CANADA               | Canada    | 0.00869856 |
| TOYOTA MOTOR CORP                  | Japan     | 0.00762924 |
| MITSUBISHI UFJ FINANCIAL GROUP INC | Japan     | 0.00730266 |
| SONY GROUP CORP                    | Japan     | 0.00721053 |

---
