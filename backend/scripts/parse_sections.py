#!/usr/bin/env python3
"""
CLI script to parse final.md and generate sections_index.json.

Usage:
    python -m backend.scripts.parse_sections
    
Or from backend directory:
    python scripts/parse_sections.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.section_parser import parse_document


def main():
    """Parse the NG12 document and generate the section index."""
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    markdown_path = project_root / "data" / "final.md"
    output_path = project_root / "data" / "sections_index.json"
    
    print(f"📄 Parsing: {markdown_path}")
    print(f"📁 Output:  {output_path}")
    print()
    
    if not markdown_path.exists():
        print(f"❌ Error: Source file not found: {markdown_path}")
        sys.exit(1)
    
    try:
        metadata = parse_document(str(markdown_path), str(output_path))
        
        print("✅ Parsing complete!")
        print()
        print("📊 Statistics:")
        print(f"   Total sections:        {metadata['total_sections']}")
        print(f"   Sections with criteria: {metadata['sections_with_criteria']}")
        print(f"   Parsed at:             {metadata['parsed_at']}")
        print()
        print(f"💾 Index saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
