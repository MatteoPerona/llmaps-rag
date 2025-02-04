import os
import csv
from pypdf import PdfReader
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define schema for extracted data
class Product(BaseModel):
    title: Optional[str] = Field(default=None, description="The title of the product")
    price: Optional[str] = Field(default=None, description="The price of the product")

class ProductList(BaseModel):
    products: List[Product]

# Initialize LLM with tool-calling feature
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(schema=ProductList)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert extraction algorithm. Extract product titles and prices accurately."),
    ("human", "{text}"),
])

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_products(text):
    """Use the structured LLM to extract product titles and prices."""
    prompt = prompt_template.invoke({"text": text})
    result = structured_llm.invoke(prompt)
    return result.products

def save_to_csv(products, csv_path):
    """Save extracted products to a CSV file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Title", "Price"])
        for product in products:
            writer.writerow([product.title, product.price])

def main():
    # Process PDF and extract products
    pdf_path = "raw-documents/Milk _ Target.pdf"
    csv_output = "output.csv"
    
    print(f"Extracting text from {pdf_path}...")
    pdf_text = extract_text_from_pdf(pdf_path)
    
    print("Extracting products using GPT-4...")
    extracted_products = extract_products(pdf_text)
    
    print(f"Saving results to {csv_output}...")
    save_to_csv(extracted_products, csv_output)
    
    print(f"Extraction complete. Data saved to {csv_output}")

if __name__ == "__main__":
    main()
