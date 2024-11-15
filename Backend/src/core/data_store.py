from typing import Dict, List, Any

# Singleton data store to be shared across modules
class DataStore:
    def __init__(self):
        self.pdf_data = {
            "text": "",
            "paragraphs": [],
            "embeddings": []
        }
        
    def has_data(self) -> bool:
        """Check if PDF data is available"""
        return len(self.pdf_data["paragraphs"]) > 0
    
    def get_paragraph_count(self) -> int:
        """Get number of paragraphs extracted"""
        return len(self.pdf_data["paragraphs"])
    
    def store_pdf_data(self, text: str, paragraphs: List[str], embeddings: List[List[float]]):
        """Store processed PDF data"""
        self.pdf_data["text"] = text
        self.pdf_data["paragraphs"] = paragraphs
        self.pdf_data["embeddings"] = embeddings
    
    def get_pdf_data(self) -> Dict[str, Any]:
        """Get stored PDF data"""
        return self.pdf_data

# Singleton instance
data_store = DataStore()
