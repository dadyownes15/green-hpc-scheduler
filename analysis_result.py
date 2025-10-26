import sys
from src.analysis import summarize_file

def main():
    if len(sys.argv) != 2:
        print("Usage: python analysis_result.py <path_to_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    try:
        summarize_file(file_path)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()