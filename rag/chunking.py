import re
from typing import List, Dict, Tuple
from langchain.text_splitter import MarkdownTextSplitter
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(
        self,
        child_chunk_size: int = 250,
        parent_chunk_size: int = 2500,
        contextual_header_size: int = 100,
        chunk_overlap: int = 50,
    ):
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.contextual_header_size = contextual_header_size
        self.chunk_overlap = chunk_overlap

        # Initialize markdown splitters
        self.parent_splitter = MarkdownTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap
        )

        self.child_splitter = MarkdownTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=chunk_overlap
        )

    def convert_to_markdown(self, text: str, document_name: str) -> str:
        """Convert plain text to markdown format."""
        # Add document title as main header
        markdown_text = f"# {document_name}\n\n"

        # Split into paragraphs and add appropriate formatting
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if paragraph looks like a heading
            if len(para) < 100 and not para.endswith("."):
                markdown_text += f"## {para}\n\n"
            else:
                markdown_text += f"{para}\n\n"

        return markdown_text

    def generate_contextual_header(
        self, document_name: str, section_header: str = ""
    ) -> str:
        """Generate contextual header for chunks."""
        if section_header:
            header = f"{document_name} > {section_header}"
        else:
            header = document_name

        # Truncate to max size
        if len(header) > self.contextual_header_size:
            header = header[: self.contextual_header_size - 3] + "..."

        return header

    def extract_section_header(self, chunk: str) -> str:
        """Extract section header from chunk if available."""
        lines = chunk.split("\n")
        for line in lines:
            line = line.strip()
            # Look for markdown headers
            if line.startswith("#"):
                return line.lstrip("#").strip()
            # Look for lines that might be headers (short, no period)
            elif len(line) < 80 and line and not line.endswith("."):
                return line

        return ""

    def create_parent_child_chunks(
        self, text: str, document_name: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Create parent and child chunks with contextual headers."""
        # Convert to markdown
        markdown_text = self.convert_to_markdown(text, document_name)

        # Create parent chunks
        parent_texts = self.parent_splitter.split_text(markdown_text)
        parent_chunks = []

        for i, parent_text in enumerate(parent_texts):
            section_header = self.extract_section_header(parent_text)
            contextual_header = self.generate_contextual_header(
                document_name, section_header
            )

            parent_chunk = {
                "text": parent_text,
                "contextual_header": contextual_header,
                "index": i,
                "type": "parent",
            }
            parent_chunks.append(parent_chunk)

        # Create child chunks from each parent chunk
        child_chunks = []
        child_index = 0

        for parent_idx, parent_chunk in enumerate(parent_chunks):
            child_texts = self.child_splitter.split_text(parent_chunk["text"])

            for child_text in child_texts:
                section_header = self.extract_section_header(child_text)
                if not section_header:
                    section_header = self.extract_section_header(parent_chunk["text"])

                contextual_header = self.generate_contextual_header(
                    document_name, section_header
                )

                child_chunk = {
                    "text": child_text,
                    "contextual_header": contextual_header,
                    "index": child_index,
                    "parent_index": parent_idx,
                    "type": "child",
                }
                child_chunks.append(child_chunk)
                child_index += 1

        logger.info(
            f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks"
        )
        return parent_chunks, child_chunks
