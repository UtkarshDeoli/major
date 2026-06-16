import json
import google.generativeai as genai
from typing import List, Dict, Any
from fastapi import HTTPException
from src.core.config import GEMINI_API_KEY
from pypdf import PdfReader

class GeminiService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Configure generation settings for more reliable JSON output
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Lower temperature for more consistent output
            top_p=0.8,
            top_k=40,
            max_output_tokens=8192,
        )
        
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config=generation_config
        )
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting text from PDF: {str(e)}"
            )
    
    async def analyze_question_papers(
        self, 
        syllabus_content: str, 
        question_papers_content: List[str]
    ) -> Dict[str, Any]:
        """Analyze question papers using Gemini 2.0 Flash"""
        
        # Combine all question papers content
        combined_question_papers = "\n\n---NEW QUESTION PAPER---\n\n".join(question_papers_content)
        
        prompt = f"""You are an expert academic analyst. Analyze the syllabus and question papers to generate insights.

SYLLABUS:
{syllabus_content[:2000]}

QUESTION PAPERS:
{combined_question_papers[:4000]}

RESPOND ONLY WITH VALID JSON IN THIS EXACT FORMAT:
{{
    "overall_summary": "Brief summary of patterns observed",
    "focus_areas": ["topic1", "topic2", "topic3", "topic4", "topic5"],
    "unit_wise_analysis": [
    There are typically 5 units in a syllabus, each with its own weightage and important topics.
    list them all and add a recommendation for each unit.
        {{
            "unit_name": "Unit 1",
            "weightage_percentage": 20.0,
            "important_topics": ["topic1", "topic2"],
            "difficulty_level": "Medium",
            "recommendation": "Study strategy"
        }}
    ],
    "question_patterns": [
        {{
            "question_type": "MCQ",
            "marks_distribution": {{"2_marks": 5, "5_marks": 3}},
            "frequency": 8,
            "examples": ["example1", "example2"]
        }}
    ],
    "sample_questions": [
        You need to generate sample questions based on the patterns observed in the syllabus and question papers. Provide 10 sample questions that reflect the types of questions typically asked in exams, such as conceptual understanding, problem-solving, and application-based questions.
        try to keep the questions relevant to the syllabus and question papers provided. 
        "Sample question 1?",
        "Sample question 2?",
        "Sample question 3?"
    ],
    "preparation_strategy": "Step by step preparation guide"
}}

IMPORTANT: Return ONLY the JSON object, no additional text or explanation."""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Check if response is valid
            if not response or not response.text:
                raise HTTPException(
                    status_code=500,
                    detail="Empty response from Gemini API"
                )
            
            response_text = response.text.strip()
            
            # Log the raw response for debugging
            print(f"Raw Gemini response: {response_text[:500]}...")
            
            # Try to extract JSON from the response
            # Sometimes Gemini adds extra text before/after JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                # If no JSON found, create a fallback response
                return self._create_fallback_analysis()
            
            json_text = response_text[start_idx:end_idx]
            
            # Parse the JSON response
            analysis_result = json.loads(json_text)
            return analysis_result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Response text: {response.text if response else 'No response'}")
            return self._create_fallback_analysis()
            
        except Exception as e:
            print(f"General error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing question papers with Gemini: {str(e)}"
            )
    
    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Create a fallback analysis when Gemini fails"""
        return {
            "overall_summary": "Unable to generate detailed analysis due to API response issues. Please ensure your documents contain clear, readable text and try again.",
            "focus_areas": [
                "Review your uploaded documents",
                "Ensure PDFs contain readable text",
                "Check document quality",
                "Retry analysis with better quality documents"
            ],
            "unit_wise_analysis": [
                {
                    "unit_name": "General Study Areas",
                    "weightage_percentage": 25.0,
                    "important_topics": ["Core concepts", "Fundamental principles", "Problem solving"],
                    "difficulty_level": "Medium",
                    "recommendation": "Focus on understanding basic concepts and practice regular problem solving"
                }
            ],
            "question_patterns": [
                {
                    "question_type": "Multiple Choice Questions",
                    "marks_distribution": {"2_marks": 10, "5_marks": 5, "10_marks": 2},
                    "frequency": 15,
                    "examples": ["Conceptual questions", "Application-based problems"]
                }
            ],
            "sample_questions": [
                "What are the fundamental concepts covered in this subject?",
                "Explain the key principles with examples.",
                "Solve problems related to core topics.",
                "Compare and contrast different approaches.",
                "Analyze the given scenario and provide solutions."
                "What are the common mistakes students make in this subject?",
            ],
            "preparation_strategy": "1. Start with basic concepts\n2. Practice previous year questions\n3. Focus on understanding rather than memorization\n4. Regular revision of important topics\n5. Time management during exams"
        }
    
    async def generate_mock_test(
        self,
        syllabus_content: str,
        notes_content: str,
        previous_papers_content: List[str],
        num_questions: int = 20,
        difficulty_level: str = "mixed"
    ) -> Dict[str, Any]:
        """Generate a mock test paper based on syllabus, notes, and previous question papers"""
        
        # Combine previous papers
        combined_previous_papers = "\n\n---PREVIOUS PAPER---\n\n".join(previous_papers_content)
        
        prompt = f"""You are an expert exam paper setter with years of experience. Your task is to create a realistic mock test paper.

INSTRUCTIONS:
1. CAREFULLY analyze the syllabus to understand course structure and learning outcomes
2. STUDY the previous year question papers to identify:
   - Question patterns and formats
   - Marks distribution (2 marks, 5 marks, 10 marks questions)
   - Frequently asked topics
   - Question difficulty progression
3. REVIEW the notes to understand depth of coverage for each topic
4. GENERATE questions that:
   - Are directly relevant to syllabus topics
   - Follow the same pattern as previous papers
   - Test conceptual understanding, application, and problem-solving
   - Have realistic difficulty levels
   - Cover different units proportionally

SYLLABUS (Use this as PRIMARY reference):
{syllabus_content[:3000]}

NOTES (Use this for topic depth):
{notes_content[:3000]}

PREVIOUS YEAR QUESTION PAPERS (Use this for pattern matching):
{combined_previous_papers[:4000]}

RESPOND ONLY WITH VALID JSON IN THIS EXACT FORMAT:
{{
    "exam_metadata": {{
        "total_marks": 100,
        "duration_minutes": 180,
        "total_questions": {num_questions},
        "pattern_match_confidence": "High/Medium/Low"
    }},
    "sections": [
        {{
            "section_name": "Section A - Multiple Choice Questions",
            "instructions": "Answer all questions. Each question carries 2 marks.",
            "total_marks": 20,
            "questions": [
                {{
                    "question_number": 1,
                    "question_text": "Which of the following...",
                    "marks": 2,
                    "difficulty": "Easy",
                    "unit": "Unit 1",
                    "topic": "Specific topic from syllabus",
                    "type": "MCQ",
                    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
                    "correct_answer": "A",
                    "explanation": "Brief explanation why this is correct",
                    "syllabus_reference": "Exact topic from syllabus",
                    "bloom_level": "Remember/Understand/Apply/Analyze/Evaluate/Create"
                }}
            ]
        }},
        {{
            "section_name": "Section B - Short Answer Questions",
            "instructions": "Answer any 5 questions. Each question carries 5 marks.",
            "total_marks": 25,
            "questions": [
                {{
                    "question_number": 11,
                    "question_text": "Explain the concept of...",
                    "marks": 5,
                    "difficulty": "Medium",
                    "unit": "Unit 2",
                    "topic": "Specific topic from syllabus",
                    "type": "Short Answer",
                    "answer_guidelines": ["Point 1", "Point 2", "Point 3"],
                    "syllabus_reference": "Exact topic from syllabus",
                    "bloom_level": "Understand/Apply"
                }}
            ]
        }},
        {{
            "section_name": "Section C - Long Answer Questions",
            "instructions": "Answer any 3 questions. Each question carries 10 marks.",
            "total_marks": 30,
            "questions": [
                {{
                    "question_number": 16,
                    "question_text": "Discuss in detail...",
                    "marks": 10,
                    "difficulty": "Hard",
                    "unit": "Unit 3",
                    "topic": "Specific topic from syllabus",
                    "type": "Long Answer",
                    "answer_guidelines": ["Introduction", "Main concepts", "Examples", "Conclusion"],
                    "syllabus_reference": "Exact topic from syllabus",
                    "bloom_level": "Analyze/Evaluate/Create"
                }}
            ]
        }}
    ],
    "marking_scheme": {{
        "mcq_negative_marking": false,
        "partial_marking": true,
        "answer_writing_tips": ["Be concise", "Use diagrams", "Give examples"]
    }},
    "topic_coverage": [
        {{
            "unit": "Unit 1",
            "topics_covered": ["topic1", "topic2"],
            "weightage_percentage": 25,
            "questions_count": 5
        }}
    ],
    "difficulty_distribution": {{
        "easy": 30,
        "medium": 50,
        "hard": 20
    }},
    "preparation_hints": [
        "Focus on topics X, Y, Z as they appear frequently in previous papers",
        "Practice numerical problems from Unit 2",
        "Understand conceptual differences between A and B"
    ]
}}

CRITICAL REQUIREMENTS:
- Every question MUST be traceable to the syllabus
- Question patterns MUST match previous year papers
- Marks distribution MUST follow previous paper trends
- NO generic questions - all must be subject-specific
- Include a mix of theoretical and practical/numerical questions
- Questions should test different cognitive levels (Bloom's Taxonomy)
- Difficulty: {difficulty_level}

Return ONLY the JSON object, no additional text."""

        try:
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                raise HTTPException(
                    status_code=500,
                    detail="Empty response from Gemini API"
                )
            
            response_text = response.text.strip()
            print(f"Mock test generation response: {response_text[:500]}...")
            
            # Extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return self._create_fallback_mock_test(num_questions)
            
            json_text = response_text[start_idx:end_idx]
            mock_test = json.loads(json_text)
            
            # Validate the structure
            if not self._validate_mock_test(mock_test):
                return self._create_fallback_mock_test(num_questions)
            
            return mock_test
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in mock test generation: {str(e)}")
            return self._create_fallback_mock_test(num_questions)
            
        except Exception as e:
            print(f"Error generating mock test: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating mock test with Gemini: {str(e)}"
            )
    
    def _validate_mock_test(self, mock_test: Dict[str, Any]) -> bool:
        """Validate mock test structure"""
        required_keys = ["exam_metadata", "sections", "marking_scheme", "topic_coverage"]
        return all(key in mock_test for key in required_keys)
    
    def _create_fallback_mock_test(self, num_questions: int) -> Dict[str, Any]:
        """Create a fallback mock test when generation fails"""
        return {
            "exam_metadata": {
                "total_marks": 100,
                "duration_minutes": 180,
                "total_questions": num_questions,
                "pattern_match_confidence": "Low"
            },
            "sections": [
                {
                    "section_name": "Section A - Multiple Choice Questions",
                    "instructions": "Answer all questions. Each question carries 2 marks.",
                    "total_marks": 20,
                    "questions": [
                        {
                            "question_number": i,
                            "question_text": f"Sample MCQ question {i}. Please regenerate with proper documents.",
                            "marks": 2,
                            "difficulty": "Medium",
                            "unit": "General",
                            "topic": "Sample topic",
                            "type": "MCQ",
                            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                            "correct_answer": "A",
                            "explanation": "Placeholder explanation",
                            "syllabus_reference": "Not available",
                            "bloom_level": "Remember"
                        }
                        for i in range(1, min(11, num_questions + 1))
                    ]
                }
            ],
            "marking_scheme": {
                "mcq_negative_marking": False,
                "partial_marking": True,
                "answer_writing_tips": ["Read questions carefully", "Manage time properly", "Review answers"]
            },
            "topic_coverage": [],
            "difficulty_distribution": {
                "easy": 33,
                "medium": 34,
                "hard": 33
            },
            "preparation_hints": [
                "Unable to generate specific hints due to document processing issues",
                "Please ensure high-quality PDFs are uploaded",
                "Retry generation with complete syllabus and previous papers"
            ]
        }

# Create global instance with proper error handling
try:
    gemini_service = GeminiService() if GEMINI_API_KEY else None
    if gemini_service:
        print("✓ Gemini service initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Gemini service: {str(e)}")
    gemini_service = None
