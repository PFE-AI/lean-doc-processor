from dotenv import load_dotenv
import os
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import base64

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Mistral AI imports
# from mistralai import Mistral

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class MistralOCRDocumentProcessor:
    """A class to handle OCR processing using Mistral's OCR capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral OCR processor.
        
        Args:
            api_key: Mistral API key. If None, will attempt to get from environment.
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key must be provided or set as MISTRAL_API_KEY environment variable")
        
        self.client = Mistral(api_key=self.api_key)
        self.ocr_model = "mistral-ocr-latest"
    
    def debug_ocr_response(self, ocr_response):
        """Debug helper to better understand the structure of OCR responses."""
        if hasattr(ocr_response, '__dict__'):
            return {k: self.debug_ocr_response(v) for k, v in ocr_response.__dict__.items() 
                    if not k.startswith('_')}
        elif isinstance(ocr_response, list):
            return [self.debug_ocr_response(item) for item in ocr_response]
        elif isinstance(ocr_response, dict):
            return {k: self.debug_ocr_response(v) for k, v in ocr_response.items()}
        else:
            return ocr_response
        
    def encode_file(self, file_path: Union[str, Path]) -> str:
        """
        Encode a file to base64.
        
        Args:
            file_path: Path to the file to encode
            
        Returns:
            Base64 encoded file content
        """
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error encoding file: {e}")
            raise
    
    def process_file(self, file_path: Union[str, Path], include_images: bool = False) -> Dict:
        """
        Process a file using Mistral OCR.
        
        Args:
            file_path: Path to the file to process
            include_images: Whether to include base64 encoded images in response
            
        Returns:
            OCR processing results
        """
        file_path = Path(file_path)
        
        # Determine file type and encoding method
        if file_path.suffix.lower() in ['.pdf']:
            file_type = "application/pdf"
            data_type = "document_url"
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            file_type = f"image/{file_path.suffix.lower().lstrip('.')}"
            data_type = "image_url"
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Encode the file
        encoded_file = self.encode_file(file_path)
        
        try:
            # Process with Mistral OCR
            ocr_response = self.client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": data_type,
                    "document_url": f"data:{file_type};base64,{encoded_file}"
                },
                include_image_base64=include_images
            )
            
            # Debug the response structure
            logger.info(f"OCR Response: {ocr_response}")
            
            return ocr_response
        except Exception as e:
            logger.error(f"Error processing document with Mistral OCR: {e}")
            raise
    
    def upload_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Upload a file for OCR processing.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            Uploaded file information
        """
        try:
            with open(file_path, "rb") as file:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": Path(file_path).name,
                        "content": file,
                    },
                    purpose="ocr"
                )
            return uploaded_file
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    def get_signed_url(self, file_id: str) -> str:
        """
        Get a signed URL for an uploaded file.
        
        Args:
            file_id: ID of the uploaded file
            
        Returns:
            Signed URL
        """
        try:
            signed_url = self.client.files.get_signed_url(file_id=file_id)
            return signed_url.url
        except Exception as e:
            logger.error(f"Error getting signed URL: {e}")
            raise
    
    def process_from_url(self, url: str, is_image: bool = False) -> Dict:
        """
        Process a document from a URL.
        
        Args:
            url: URL of the document to process
            is_image: Whether the URL points to an image
            
        Returns:
            OCR processing results
        """
        try:
            ocr_response = self.client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "image_url" if is_image else "document_url",
                    "document_url": url
                }
            )
            return ocr_response
        except Exception as e:
            logger.error(f"Error processing document from URL: {e}")
            raise


class MistralLangChainQAPipeline:
    """A class to handle document Q&A using Mistral and LangChain."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "mistral-small-latest"):
        """
        Initialize the Q&A pipeline.
        
        Args:
            api_key: Mistral API key. If None, will attempt to get from environment.
            model_name: Mistral model name to use for Q&A
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key must be provided or set as MISTRAL_API_KEY environment variable")
        
        self.client = Mistral(api_key=self.api_key)
        self.model_name = model_name
        self.ocr_processor = MistralOCRDocumentProcessor(api_key=self.api_key)
        
    def _create_langchain_documents(self, ocr_result) -> List[Document]:
        """
        Convert OCR results to LangChain Document objects.
        
        Args:
            ocr_result: OCR processing results
            
        Returns:
            List of LangChain Document objects
        """
        # Based on the error and the logged structure, we need to properly adapt to the actual response format
        documents = []
        
        try:
            # Debug the structure
            debug_struct = self.ocr_processor.debug_ocr_response(ocr_result)
            logger.info(f"Parsed OCR result structure: {debug_struct}")
            
            # The Mistral OCR format seems to have pages as a list of objects
            if hasattr(ocr_result, 'pages') and ocr_result.pages:
                for i, page in enumerate(ocr_result.pages):
                    # Check if page has markdown content
                    if hasattr(page, 'markdown'):
                        # Create a Document with the page's markdown content
                        document = Document(
                            page_content=page.markdown,
                            metadata={
                                "source": "mistral_ocr",
                                "page_number": i,
                                "page_dimensions": {
                                    "width": page.dimensions.width if hasattr(page, 'dimensions') and hasattr(page.dimensions, 'width') else None,
                                    "height": page.dimensions.height if hasattr(page, 'dimensions') and hasattr(page.dimensions, 'height') else None,
                                    "dpi": page.dimensions.dpi if hasattr(page, 'dimensions') and hasattr(page.dimensions, 'dpi') else None
                                } if hasattr(page, 'dimensions') else {}
                            }
                        )
                        documents.append(document)
            else:
                logger.warning("No pages found in OCR result")
                
                # Fallback approach - try to extract any text we can find
                if hasattr(ocr_result, '__dict__'):
                    for key, value in ocr_result.__dict__.items():
                        if isinstance(value, str) and len(value) > 100:  # Likely to be text content
                            document = Document(
                                page_content=value,
                                metadata={"source": "mistral_ocr", "extraction_method": "fallback"}
                            )
                            documents.append(document)
                            break
            
            # If we still don't have any documents, create one with a warning
            if not documents:
                logger.warning("Creating fallback document with warning")
                document = Document(
                    page_content="Could not extract meaningful text from the document.",
                    metadata={"source": "mistral_ocr", "extraction_status": "failed"}
                )
                documents.append(document)
                
            return documents
            
        except Exception as e:
            logger.error(f"Error creating LangChain documents: {e}")
            # Return at least an empty document so the chain doesn't break
            return [Document(
                page_content="Error extracting text from document.",
                metadata={"source": "mistral_ocr", "error": str(e)}
            )]
    
    def _split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into chunks for processing.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        return text_splitter.split_documents(documents)
    
    def answer_question_direct(self, document_url: str, question: str, is_image: bool = False) -> str:
        """
        Answer a question directly using Mistral's document understanding capability.
        
        Args:
            document_url: URL of the document to process
            question: Question to answer
            is_image: Whether the URL points to an image
            
        Returns:
            Answer to the question
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "document_url" if not is_image else "image_url",
                            "document_url": document_url
                        }
                    ]
                }
            ]
            
            chat_response = self.client.chat.complete(
                model=self.model_name,
                messages=messages
            )
            
            return chat_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error answering question directly: {e}")
            raise
    
    def answer_question_with_langchain(self, 
                                      document_path: Union[str, Path], 
                                      question: str, 
                                      use_retrieval: bool = True,
                                      chunk_size: int = 1000,
                                      chunk_overlap: int = 200) -> str:
        """
        Process a document and answer a question using LangChain pipeline.
        
        Args:
            document_path: Path to the document to process
            question: Question to answer
            use_retrieval: Whether to use retrieval-based QA
            chunk_size: Size of document chunks for splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            Answer to the question
        """
        try:
            # Process the document with OCR
            ocr_result = self.ocr_processor.process_file(document_path)
            
            # Print OCR result structure for debugging
            logger.info(f"OCR result structure: {ocr_result}")
            
            # Convert OCR results to LangChain documents
            documents = self._create_langchain_documents(ocr_result)
            
            if not documents:
                logger.warning("No documents created from OCR result")
                return "Could not extract text from the document to answer your question."
            
            # Split documents into chunks
            document_chunks = self._split_documents(documents, chunk_size, chunk_overlap)
            
            if use_retrieval:
                # Create embeddings using Mistral's API
                # Note: For this example, we'll use a simple approach.
                # In production, you'd want to use a proper embedding model.
                
                # Create a retriever
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                embeddings = HuggingFaceEmbeddings()
                vector_store = Chroma.from_documents(document_chunks, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Define the prompt template
                template = """
                You are an AI assistant tasked with answering questions based on the provided document content.
                Use the following pieces of document context to answer the question.
                If you don't know the answer, just say you don't know. Don't try to make up an answer.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                # Create and run a retrieval chain
                def format_docs(docs):
                    return "\n\n".join([doc.page_content for doc in docs])
                
                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | self._create_mistral_llm()
                    | StrOutputParser()
                )
                
                return chain.invoke(question)
            else:
                # Direct QA without retrieval
                qa_prompt = PromptTemplate(
                    template="""
                    You are an AI assistant tasked with answering questions based on the provided document content.
                    Document content:
                    {document_content}
                    
                    Question: {question}
                    
                    Answer:
                    """,
                    input_variables=["document_content", "question"]
                )
                
                document_content = "\n\n".join([doc.page_content for doc in document_chunks])
                
                chain = (
                    qa_prompt.format(document_content=document_content, question=question)
                    | self._create_mistral_llm()
                    | StrOutputParser()
                )
                
                return chain.invoke({})
        except Exception as e:
            logger.error(f"Error answering question with LangChain: {e}")
            raise
    
    def _create_mistral_llm(self):
        """Create a LangChain-compatible Mistral LLM."""
        from langchain_core.language_models.chat_models import ChatOpenAI
        from langchain_mistralai.chat_models import ChatMistralAI
        
        return ChatMistralAI(
            model_name=self.model_name,
            mistral_api_key=self.api_key,
            temperature=0.1,
            max_tokens=1024
        )


# Example usage
def example_usage():
    """Example usage of the Mistral OCR LangChain Pipeline."""
    # Set up the pipeline
    api_key = os.getenv("MISTRAL_API_KEY")
    pipeline = MistralLangChainQAPipeline(api_key=api_key)
    
    # Process a local PDF and answer a question using retrieval
    pdf_path = "../data/generated_data/bill_payments_80mm/bill_payment_receipt_821952_80mm.pdf"
    question = "What are the key points of the document?"
    
    print(f"Processing document: {pdf_path}")
    print(f"Question: {question}")
    
    try:
        answer = pipeline.answer_question_with_langchain(
            document_path=pdf_path,
            question=question,
            use_retrieval=True
        )
        
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error processing document: {e}")
    
    # Example of direct document Q&A from URL
    document_url = "https://arxiv.org/pdf/1805.04770"
    question_url = "What is the main contribution of this paper?"
    
    print(f"\nProcessing document from URL: {document_url}")
    print(f"Question: {question_url}")
    
    try:
        answer_url = pipeline.answer_question_direct(
            document_url=document_url,
            question=question_url
        )
        
        print(f"Answer: {answer_url}")
    except Exception as e:
        print(f"Error processing document from URL: {e}")


if __name__ == "__main__":
    example_usage()