"""
Section Parser for NG12 Document.

Parses final.md into structured sections with metadata for retrieval and pathway UI.
Run once via CLI to generate sections_index.json.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup


@dataclass
class Criterion:
    """A single criterion for pathway evaluation."""
    field: str          # "age", "smoking_status", "symptom", "finding"
    operator: str       # ">=", "==", "in", "has_any", "has_all"
    value: any          # 40, True, ["cough", "fatigue"]
    label: str          # Human-readable: "Age 40 or over"


@dataclass
class CriteriaGroup:
    """A group of criteria with AND/OR logic."""
    operator: str       # "AND", "OR"
    criteria: list[dict] = field(default_factory=list)  # List of Criterion as dicts


@dataclass
class CriteriaSpec:
    """Pre-parsed criteria specification for pathway UI."""
    recommendation_id: str
    description: str
    criteria_groups: list[dict] = field(default_factory=list)  # List of CriteriaGroup as dicts
    action: str = ""


@dataclass
class ParsedSection:
    """A parsed section from the NG12 document."""
    id: str                     # "1.1.2" or "terms-unexplained"
    header: str                 # Full header text
    header_path: list[str]      # Breadcrumb: ["1.1 Lung", "Lung cancer", "1.1.2"]
    content: str                # Verbatim text
    level: int                  # Header depth (1-4)
    start_line: int
    end_line: int
    section_type: str           # "recommendation", "definition", "symptom_table", "guidance", "overview"
    has_criteria: bool = False  # True if contains age/symptom criteria
    criteria_spec: Optional[dict] = None  # CriteriaSpec as dict for pathway UI
    cancer_site: Optional[str] = None     # "lung", "colorectal", etc.
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SectionParser:
    """Parser for NG12 document into structured sections."""
    
    # Regex patterns
    HEADER_PATTERN = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    RECOMMENDATION_ID_PATTERN = re.compile(r'^(\d+\.\d+\.\d+)\s+(.+)', re.DOTALL)
    AGE_PATTERN = re.compile(r'aged?\s*(\d+)\s*(?:and\s*over|or\s*over|\+|years?\s*and\s*over)', re.IGNORECASE)
    SMOKING_PATTERN = re.compile(r'(?:have\s*)?ever\s*smoked', re.IGNORECASE)
    SYMPTOM_COUNT_PATTERN = re.compile(r'(\d+)\s*or\s*more\s*(?:of\s*the\s*following)?(?:\s*unexplained)?\s*symptoms?', re.IGNORECASE)
    
    # Cancer site mapping
    CANCER_SITES = {
        "1.1": "lung",
        "1.2": "upper_gi",
        "1.3": "lower_gi",
        "1.4": "breast",
        "1.5": "gynaecological",
        "1.6": "urological",
        "1.7": "skin",
        "1.8": "head_neck",
        "1.9": "brain_cns",
        "1.10": "haematological",
        "1.11": "sarcoma",
        "1.12": "childhood",
        "1.13": "non_site_specific",
    }
    
    def __init__(self):
        self.sections: list[ParsedSection] = []
        self.header_stack: list[tuple[int, str]] = []  # (level, header_text)
    
    def parse(self, markdown_path: str) -> list[ParsedSection]:
        """Parse the markdown document into sections."""
        path = Path(markdown_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {markdown_path}")
        
        content = path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        self.sections = []
        self.header_stack = []
        
        current_section_start = 0
        current_header = ""
        current_level = 0
        current_content_lines = []
        
        for i, line in enumerate(lines):
            line_num = i + 1  # 1-indexed
            header_match = self.HEADER_PATTERN.match(line)
            
            if header_match:
                # Save previous section if exists
                if current_header:
                    self._save_section(
                        header=current_header,
                        level=current_level,
                        content_lines=current_content_lines,
                        start_line=current_section_start,
                        end_line=line_num - 1
                    )
                
                # Start new section
                hashes = header_match.group(1)
                header_text = header_match.group(2).strip()
                current_level = len(hashes)
                current_header = header_text
                current_section_start = line_num
                current_content_lines = []
                
                # Update header stack for breadcrumb
                self._update_header_stack(current_level, header_text)
            else:
                current_content_lines.append(line)
        
        # Save final section
        if current_header:
            self._save_section(
                header=current_header,
                level=current_level,
                content_lines=current_content_lines,
                start_line=current_section_start,
                end_line=len(lines)
            )
        
        # Post-process: extract recommendations from content
        self._extract_inline_recommendations(lines)
        
        return self.sections
    
    def _update_header_stack(self, level: int, header_text: str):
        """Update the header stack for building breadcrumbs."""
        # Remove all headers at same or deeper level
        while self.header_stack and self.header_stack[-1][0] >= level:
            self.header_stack.pop()
        self.header_stack.append((level, header_text))
    
    def _get_header_path(self) -> list[str]:
        """Get current breadcrumb path from header stack."""
        return [h[1] for h in self.header_stack]
    
    def _save_section(self, header: str, level: int, content_lines: list[str], 
                      start_line: int, end_line: int):
        """Save a parsed section."""
        content = '\n'.join(content_lines).strip()
        
        # Determine section ID
        section_id = self._generate_section_id(header, content)
        
        # Determine section type
        section_type = self._determine_section_type(header, content)
        
        # Determine cancer site
        cancer_site = self._determine_cancer_site(section_id, self._get_header_path())
        
        # Check for criteria
        has_criteria, criteria_spec = self._extract_criteria(section_id, header, content)
        
        section = ParsedSection(
            id=section_id,
            header=header,
            header_path=self._get_header_path().copy(),
            content=content,
            level=level,
            start_line=start_line,
            end_line=end_line,
            section_type=section_type,
            has_criteria=has_criteria,
            criteria_spec=criteria_spec,
            cancer_site=cancer_site
        )
        
        self.sections.append(section)
    
    def _generate_section_id(self, header: str, content: str) -> str:
        """Generate a unique section ID."""
        # Check if header starts with recommendation number
        rec_match = self.RECOMMENDATION_ID_PATTERN.match(header)
        if rec_match:
            return rec_match.group(1)
        
        # Check for numbered section (e.g., "1.1 Lung and pleural cancers")
        numbered_match = re.match(r'^(\d+\.\d+)\s+', header)
        if numbered_match:
            return numbered_match.group(1)
        
        # Generate slug from header
        slug = re.sub(r'[^a-z0-9]+', '-', header.lower()).strip('-')
        return slug[:50]  # Limit length
    
    def _determine_section_type(self, header: str, content: str) -> str:
        """Determine the type of section."""
        header_lower = header.lower()
        
        # Check for recommendation (numbered)
        if re.match(r'^\d+\.\d+\.\d+\s+', header):
            return "recommendation"
        
        # Check for HTML table (symptom table)
        if '<table' in content.lower():
            return "symptom_table"
        
        # Check for terms/definitions section
        header_path_str = ' > '.join(self._get_header_path()).lower()
        if 'terms' in header_path_str or 'definition' in header_lower:
            return "definition"
        
        # Check for research section
        if 'research' in header_lower or 'rationale' in header_lower:
            return "research"
        
        # Check for overview/introduction
        if any(kw in header_lower for kw in ['overview', 'introduction', 'contents', 'responsibility']):
            return "overview"
        
        # Default to guidance
        return "guidance"
    
    def _determine_cancer_site(self, section_id: str, header_path: list[str]) -> Optional[str]:
        """Determine the cancer site from section context."""
        # Check section ID prefix
        for prefix, site in self.CANCER_SITES.items():
            if section_id.startswith(prefix):
                return site
        
        # Check header path for cancer site mentions
        path_str = ' '.join(header_path).lower()
        site_keywords = {
            'lung': 'lung',
            'pleural': 'lung',
            'mesothelioma': 'lung',
            'oesophageal': 'upper_gi',
            'stomach': 'upper_gi',
            'pancreatic': 'upper_gi',
            'liver': 'upper_gi',
            'gallbladder': 'upper_gi',
            'colorectal': 'lower_gi',
            'bowel': 'lower_gi',
            'anal': 'lower_gi',
            'breast': 'breast',
            'ovarian': 'gynaecological',
            'endometrial': 'gynaecological',
            'cervical': 'gynaecological',
            'vulval': 'gynaecological',
            'vaginal': 'gynaecological',
            'prostate': 'urological',
            'bladder': 'urological',
            'renal': 'urological',
            'testicular': 'urological',
            'penile': 'urological',
            'skin': 'skin',
            'melanoma': 'skin',
            'laryngeal': 'head_neck',
            'thyroid': 'head_neck',
            'oral': 'head_neck',
            'brain': 'brain_cns',
            'leukaemia': 'haematological',
            'lymphoma': 'haematological',
            'myeloma': 'haematological',
            'sarcoma': 'sarcoma',
            'neuroblastoma': 'childhood',
            'retinoblastoma': 'childhood',
            'wilms': 'childhood',
        }
        
        for keyword, site in site_keywords.items():
            if keyword in path_str:
                return site
        
        return None
    
    def _extract_criteria(self, section_id: str, header: str, content: str) -> tuple[bool, Optional[dict]]:
        """Extract criteria from a recommendation section."""
        full_text = f"{header}\n{content}"
        
        # Only extract criteria from recommendation sections
        if not re.match(r'^\d+\.\d+\.\d+$', section_id):
            return False, None
        
        # Exclude non-clinical sections (patient info, safety netting, diagnostic process, research)
        # These sections have bullet points but they're not clinical referral criteria
        non_clinical_prefixes = ('1.14.', '1.15.', '1.16.')
        if section_id.startswith(non_clinical_prefixes):
            return False, None
        
        # Exclude sections about "information" or "should cover" - these are process requirements
        header_lower = header.lower()
        if any(phrase in header_lower for phrase in [
            'information given',
            'should cover',
            'information and support',
            'discuss with people',
            'explain to people',
        ]):
            return False, None
        
        criteria_groups = []
        
        # Extract age criterion
        age_match = self.AGE_PATTERN.search(full_text)
        age_value = int(age_match.group(1)) if age_match else None
        
        # Extract smoking criterion
        has_smoking = bool(self.SMOKING_PATTERN.search(full_text))
        
        # Extract symptom count
        symptom_count_match = self.SYMPTOM_COUNT_PATTERN.search(full_text)
        symptom_threshold = int(symptom_count_match.group(1)) if symptom_count_match else None
        
        # Extract symptom list (bullet points)
        symptoms = self._extract_symptom_list(content)
        
        # Build criteria groups based on patterns found
        if age_value or has_smoking or symptoms:
            criteria = []
            
            if age_value:
                criteria.append({
                    "field": "age",
                    "operator": ">=",
                    "value": age_value,
                    "label": f"Aged {age_value} or over"
                })
            
            if has_smoking:
                criteria.append({
                    "field": "smoking",
                    "operator": "==",
                    "value": True,
                    "label": "Has ever smoked"
                })
            
            if symptoms:
                operator = "has_any"
                threshold_label = ""
                if symptom_threshold and symptom_threshold > 1:
                    operator = f"has_{symptom_threshold}_or_more"
                    threshold_label = f" ({symptom_threshold}+ required)"
                
                criteria.append({
                    "field": "symptoms",
                    "operator": operator,
                    "value": symptoms,
                    "label": f"Has unexplained symptoms{threshold_label}"
                })
            
            if criteria:
                criteria_groups.append({
                    "operator": "AND",
                    "criteria": criteria
                })
        
        has_criteria = len(criteria_groups) > 0
        
        if has_criteria:
            # Extract action from header/content
            action = self._extract_action(header, content)
            
            criteria_spec = {
                "recommendation_id": section_id,
                "description": header[:100],
                "criteria_groups": criteria_groups,
                "action": action
            }
            return True, criteria_spec
        
        return False, None
    
    def _extract_symptom_list(self, content: str) -> list[str]:
        """Extract symptom list from bullet points, cleaning up non-symptom text."""
        symptoms = []
        
        # Match bullet points (*, -, •)
        bullet_pattern = re.compile(r'^[\s]*[*\-•]\s*(.+?)(?:\s*\[|\s*$)', re.MULTILINE)
        
        # Phrases that indicate this is not a symptom but instructions/conditions
        non_symptom_phrases = [
            'presenting with', 'for the first time', 'assess her', 'assess him',
            'investigate if', 'clinical causes', 'who:', 'who are', 'and:',
            'consider', 'refer', 'offer', 'discuss', 'explain', 'arrange',
            'should', 'must', 'need to', 'required', 'appropriate',
        ]
        
        for match in bullet_pattern.finditer(content):
            symptom = match.group(1).strip()
            
            # Clean up the symptom text
            symptom = re.sub(r'\s*\[.*?\]', '', symptom)  # Remove [2015] etc
            symptom = re.sub(r'</?u>', '', symptom)  # Remove <u> tags
            
            # Remove trailing conjunctions and punctuation
            symptom = re.sub(r'\s+(or|and|who|with|that|which)\s*:?\s*$', '', symptom, flags=re.IGNORECASE)
            symptom = symptom.rstrip('.,;:')
            
            # Skip if it contains non-symptom phrases
            symptom_lower = symptom.lower()
            if any(phrase in symptom_lower for phrase in non_symptom_phrases):
                continue
            
            # Skip if too long (likely a sentence, not a symptom)
            if len(symptom) > 60:
                continue
            
            # Skip if too short
            if len(symptom) < 3:
                continue
            
            # Skip duplicates (case-insensitive)
            if symptom_lower not in [s.lower() for s in symptoms]:
                symptoms.append(symptom.lower())
        
        return symptoms
    
    def _extract_action(self, header: str, content: str) -> str:
        """Extract the recommended action from the section."""
        full_text = f"{header}\n{content}"
        
        # Common action patterns
        action_patterns = [
            r'(Refer\s+(?:people\s+)?using\s+a\s+suspected\s+cancer\s+pathway\s+referral)',
            r'(Offer\s+(?:an?\s+)?(?:urgent|very urgent)\s+[^.]+)',
            r'(Consider\s+(?:an?\s+)?(?:urgent|very urgent)?\s*(?:referral|[^.]+))',
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: first sentence of header
        return header.split('.')[0] if '.' in header else header
    
    def _extract_inline_recommendations(self, lines: list[str]):
        """Extract numbered recommendations that appear inline within sections."""
        # Find recommendations that start with X.X.X pattern in the content
        rec_pattern = re.compile(r'^(\d+\.\d+\.\d+)\s+(.+)', re.MULTILINE)
        
        full_content = '\n'.join(lines)
        
        for match in rec_pattern.finditer(full_content):
            rec_id = match.group(1)
            
            # Check if we already have this recommendation as a section
            existing = [s for s in self.sections if s.id == rec_id]
            if existing:
                continue
            
            # Find the line number
            start_pos = match.start()
            line_num = full_content[:start_pos].count('\n') + 1
            
            # Extract the full recommendation text (until next numbered recommendation or header)
            remaining = full_content[match.start():]
            end_match = re.search(r'\n(?:\d+\.\d+\.\d+\s+|#{1,4}\s+)', remaining[1:])
            if end_match:
                rec_text = remaining[:end_match.start() + 1]
            else:
                # Take until double newline
                end_match = re.search(r'\n\n', remaining)
                rec_text = remaining[:end_match.start()] if end_match else remaining[:500]
            
            end_line = line_num + rec_text.count('\n')
            
            # Determine cancer site from ID
            cancer_site = None
            for prefix, site in self.CANCER_SITES.items():
                if rec_id.startswith(prefix):
                    cancer_site = site
                    break
            
            # Extract criteria
            has_criteria, criteria_spec = self._extract_criteria(rec_id, rec_text.split('\n')[0], rec_text)
            
            section = ParsedSection(
                id=rec_id,
                header=rec_text.split('\n')[0].strip(),
                header_path=[f"Recommendation {rec_id}"],
                content=rec_text.strip(),
                level=5,  # Inline recommendation
                start_line=line_num,
                end_line=end_line,
                section_type="recommendation",
                has_criteria=has_criteria,
                criteria_spec=criteria_spec,
                cancer_site=cancer_site
            )
            
            self.sections.append(section)
    
    def parse_tables(self, content: str) -> list[dict]:
        """Parse HTML tables from content into structured data."""
        tables = []
        soup = BeautifulSoup(content, 'html.parser')
        
        for table in soup.find_all('table'):
            headers = []
            rows = []
            
            # Get headers
            thead = table.find('thead')
            if thead:
                for th in thead.find_all('th'):
                    headers.append(th.get_text(strip=True))
            
            # Get rows
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    row = []
                    for td in tr.find_all('td'):
                        row.append(td.get_text(strip=True))
                    if row:
                        rows.append(row)
            
            if headers and rows:
                tables.append({
                    'headers': headers,
                    'rows': rows
                })
        
        return tables
    
    def to_index(self, source_path: str) -> dict:
        """Convert parsed sections to index format for JSON export."""
        return {
            "metadata": {
                "source": source_path,
                "parsed_at": datetime.utcnow().isoformat() + "Z",
                "total_sections": len(self.sections),
                "sections_with_criteria": sum(1 for s in self.sections if s.has_criteria)
            },
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def save_index(self, output_path: str, source_path: str):
        """Save the parsed sections to a JSON index file."""
        index = self.to_index(source_path)
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        return index["metadata"]


def parse_document(markdown_path: str, output_path: str) -> dict:
    """
    Parse a markdown document and save the index.
    
    Args:
        markdown_path: Path to the source markdown file
        output_path: Path to save the JSON index
        
    Returns:
        Metadata about the parsed document
    """
    parser = SectionParser()
    parser.parse(markdown_path)
    return parser.save_index(output_path, markdown_path)
