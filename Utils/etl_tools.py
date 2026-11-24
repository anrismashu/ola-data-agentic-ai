import os   
import requests
import pandas as pd

class ETLTools:
    
    
    def __init__(self):
        pass
    
    def extract_data(self,url:str,output_folder:str,format:str):
        """
        This tool extracts data from the api url and load it into the desired
        location destination output_folder. 
        Args:
            url (str): The API URL to extract data from.
            output_folder (str): The destination folder to save the extracted data.
        Returns:
            str: a message indicating the success or failure of the extraction process.
        """     
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_folder = os.path.join(project_root, output_folder)     
        try :
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()  # Assuming the API returns JSON data            
            filename = os.path.join(output_folder, f"extracted_data.{format}")
            os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist          
            df = pd.json_normalize(data['results'])  # Convert JSON data to a DataFrame
            
            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient='records', lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported format: {format}. Please choose 'csv', 'json', or 'parquet'."       
            return f"Data extracted and saved to {filename} successfully."
        except requests.exceptions.RequestException as e:
            return f"Error occurred while extracting data: {e}"
        
    def transform_load_context(self, file_path:str):
        """
        This tool transforms and loads data from the specified file path to the desired
        location destination output_folder. 
        Args:
            file_path (str): The path to the input file to be transformed and loaded.
            output_folder (str): The destination folder to save the transformed data.
            output_format (str): The format to save the transformed data (csv, json, parquet).
        Returns:
            str: a message indicating the success or failure of the transformation and loading process.
        """   
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".json":
            df = pd.read_json(file_path, lines=True)
        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_extension}. Please provide a valid CSV, JSON, or Parquet file."
        
        top_3_rows = str(df.head(3))  # Convert the top 3 rows to a string representation
        return top_3_rows  # Return the string representation of the top 3 rows
    
    def execute_code(self, code:str):
        """
        This tool executes the provided code and returns the output.
        Args:
            code (str): The code to be executed.
        Returns:
            str: The output of the executed code or an error message if execution fails.
        """
        try:
            exec(code)
            return "Code executed successfully."
        except Exception as e:
            return f"Error occurred while executing code: {e}"


if __name__ == "__main__": 
    obj = ETLTools()
    path = "C:\EveryPython\Ola_Data_\Ola_Data_Agentic_Ai\data\output\extracted_data.csv"
    print(obj.transform_load_context(path)) 